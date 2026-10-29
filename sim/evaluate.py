"""Policy rollouts in sim under the evaluation conditions (1tasks_scene.md, "Plan").

    python -m sim.evaluate outputs/s1a_act/checkpoints/last/pretrained_model --variants S1:A
    python -m sim.evaluate CKPT --variants S1:A --condition eval_colours --episodes 50 --videos 3

Each rollout starts from `reset` on an EVAL_SEEDS seed (the same seeds for every variant and model),
renders the three cameras as the recorder does, and sends the policy's command to the actuators
held for CONTROL_EVERY steps (50 Hz, as recorded). No demonstration noise: that is the expert's.
Rollouts don't stop at success; each runs HORIZON frames, the task's longest demonstration (with
its 2 s hold) plus 2 s, as CALVIN gives each subtask a fixed horizon, and `success` is checked at
the end. Two more rates for the scenes where a primitive ends as a composite continues (1tasks_scene.md,
"Horizon and stopping"):
    stop rate (S1, S3)       success, and in the last 2 s no arm joint moves > HOLD_STILL and the cube
                             and drawer move < PUSHED (the expert's hold check)
    continue rate (T1, T2)   once the first step is done (T1: drawer open as S1 requires; T2: cube upright
                             in the drawer), the second arm leaves home (a joint > IDLE_MOVE from its start)
                             before the horizon ends
A policy trained with the variant input (sim/dataset_view.py --variant-input) gets the variant's
one-hot vector as observation.environment_state; a language-conditioned one reads the command.

Conditions (one held-out factor each, as LIBERO-PRO / LIBERO-Plus report each perturbation):
    train          train colours, train phrasings (the headline number)
    eval_colours   held-out colours
    eval_wording   held-out phrasings (language-conditioned policies only)

Writes `<out>/<condition>/episodes.jsonl` (one line per rollout: variant, seed, success, checks,
stopped or continued) and `summary.json`, plus overview videos of the first --videos rollouts per variant.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from sim import episode as E
from sim import randomize as RZ
from sim import record as R
from sim import tasks as T
from sim.dataset_view import ENV
from sim.scene import ARM_NAME, CONTROL_EVERY
from sim.skills import HOLD_S, HOLD_STILL

# Longest demonstration per task in frames, with the hold (2data_collection.md, "Hook"; S2 over both
# start kinds), plus 2 s.
DEMO_FRAMES = {"S1": 524, "S2": 520, "S3": 644, "S4": 586, "T1": 1020, "T2": 1056}
HORIZON = {task: n + 2 * R.FPS for task, n in DEMO_FRAMES.items()}
STOP_TASKS = ("S1", "S3")
# When a composite's first step is done (the end of S1 or S3), from the simulator alone.
FIRST_STEP_DONE = {"T1": lambda sim: sim.drawer_opening() > T.DRAWER_OPEN - 0.01,
                   "T2": E.cube_upright_in_drawer}
CONDITIONS = {"train": RZ.Randomization(),
              "eval_colours": RZ.Randomization(colours_split="eval"),
              "eval_wording": RZ.Randomization(wording_split="eval")}


def load_policy(path: Path, device: str):
    """A LeRobot checkpoint (`pretrained_model` directory) with its pre- and post-processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    # Render at the training size: ACT/DP keep the dataset's image shape (C, H, W); SmolVLA keeps its own camera
    # names (the saved preprocessor renames ours) and resizes inside, so it gets the recorder's size.
    overview = cfg.input_features.get("observation.images.overview")
    policy.image_size = tuple(overview.shape[1:]) if overview else (R.HEIGHT, R.WIDTH)
    policy.variant_input = ENV in cfg.input_features
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def observation(sim, qadr, renderers, command: str, variant_input=None) -> dict:
    """One frame as the recorder writes it, batched: state, the three cameras (channel first, 0-1), task,
    and the variant vector if the policy takes one."""
    obs = {"observation.state": torch.from_numpy(sim.data.qpos[qadr].astype(np.float32))[None], "task": [command]}
    if variant_input is not None:
        obs[ENV] = torch.from_numpy(variant_input)[None]
    for key, cam in R.CAMERAS.items():
        renderers[key].update_scene(sim.data, camera=cam)
        img = torch.from_numpy(renderers[key].render().copy())
        obs[f"observation.images.{key}"] = img.permute(2, 0, 1)[None].float() / 255
    return obs


def rollout(policy, pre, post, variant, seed: int, rand: RZ.Randomization, video: bool = False) -> dict:
    sim, start = E.reset(variant, seed, dataclasses.replace(rand, demo_noise=0.0))
    sim.control_every = 1  # ctrl is set once per frame and held for CONTROL_EVERY steps below
    qadr = R.joint_qpos(sim)
    arm_q = np.concatenate([sim.arms[a].qadr for a in sim.arms])
    size = getattr(policy, "image_size", (R.HEIGHT, R.WIDTH))
    renderers = {key: mujoco.Renderer(sim.model, *size) for key in R.CAMERAS}
    variant_input = RZ.variant_vector(*variant) if getattr(policy, "variant_input", False) else None
    second = sim.arms[ARM_NAME[E.other(variant[1])]].qadr
    first_done, second_left = False, 0.0  # rad, the second arm's largest joint distance from its start after the first step
    policy.reset()
    hold = int(HOLD_S * R.FPS)
    q_tail, items_tail, frames = [], [], []
    for _ in range(HORIZON[variant[0]]):
        obs = observation(sim, qadr, renderers, start["command"], variant_input)
        if video:
            frames.append((obs["observation.images.overview"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        sim.data.ctrl[:] = action[0].cpu().numpy()
        sim.step(CONTROL_EVERY)
        q_tail = (q_tail + [sim.data.qpos[arm_q].copy()])[-hold:]
        items_tail = (items_tail + [np.concatenate([sim.body_pos("cube"), [sim.drawer_opening()]])])[-hold:]
        if variant[0] in FIRST_STEP_DONE:
            first_done = first_done or bool(FIRST_STEP_DONE[variant[0]](sim))
            if first_done:
                q0 = sim.monitor.q0[ARM_NAME[E.other(variant[1])]]
                second_left = max(second_left, float(np.abs(sim.data.qpos[second] - q0).max()))
    for r in renderers.values():
        r.close()
    ok, checks = E.success(sim, start)
    q_tail, items_tail = np.array(q_tail), np.array(items_tail)
    still = (np.abs(q_tail - q_tail[0]).max() < HOLD_STILL
             and np.linalg.norm(items_tail[:, :3] - items_tail[0, :3], axis=1).max() < E.PUSHED
             and np.abs(items_tail[:, 3] - items_tail[0, 3]).max() < E.PUSHED)
    result = {"variant": list(variant), "seed": seed, "command": start["command"],
              "cube_colour": start["scene"].get("cube_colour"), "success": ok, "checks": checks}
    if variant[0] in STOP_TASKS:
        result["stopped"] = bool(ok and still)
    if variant[0] in FIRST_STEP_DONE:
        result["continued"] = bool(first_done and second_left > E.IDLE_MOVE)
    return result, frames


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", type=Path, help="a pretrained_model directory")
    p.add_argument("--variants", default="all", help="as sim.record, e.g. S1:A or S4:B:left")
    p.add_argument("--condition", choices=CONDITIONS, default="train")
    p.add_argument("--episodes", type=int, default=50, help="rollouts per variant (the first EVAL_SEEDS)")
    p.add_argument("--videos", type=int, default=0, help="save the overview of the first N rollouts per variant")
    p.add_argument("--out", type=Path, help="default: <checkpoint>/../eval")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    out = (a.out or a.checkpoint.parent / "eval") / a.condition
    out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(a.checkpoint, a.device)
    rand = CONDITIONS[a.condition]
    summary = {"checkpoint": str(a.checkpoint), "condition": a.condition, "variants": {}}
    with open(out / "episodes.jsonl", "w") as log:
        for variant in R.parse_variants(a.variants):
            name = ":".join(x for x in variant if x)
            results = []
            for k, seed in enumerate(RZ.EVAL_SEEDS[: a.episodes]):
                t = time.time()
                r, frames = rollout(policy, pre, post, variant, seed, rand, video=k < a.videos)
                if frames:
                    from lerobot.utils.io_utils import write_video
                    write_video(out / f"{name.replace(':', '_')}_{seed}_{'ok' if r['success'] else 'fail'}.mp4", frames, R.FPS)
                results.append(r)
                log.write(R.to_json(r) + "\n")
                log.flush()
                failed = [c for c, v in r["checks"].items() if not v]
                print(f"{name} seed {seed}: {'success' if r['success'] else 'fail ' + ','.join(failed)}"
                      f"{'' if 'stopped' not in r else ', stopped' if r['stopped'] else ', did not stop'}"
                      f"{'' if 'continued' not in r else ', continued' if r['continued'] else ', did not continue'}"
                      f" ({time.time() - t:.0f} s)",
                      flush=True)
            s = {"episodes": len(results), "success": float(np.mean([r["success"] for r in results])),
                 "failed_checks": {c: sum(not r["checks"][c] for r in results) for c in results[0]["checks"]}}
            if variant[0] in STOP_TASKS:
                s["stop"] = float(np.mean([r["stopped"] for r in results]))
            if variant[0] in FIRST_STEP_DONE:
                s["continue"] = float(np.mean([r["continued"] for r in results]))
            summary["variants"][name] = s
            print(f"{name}: success {s['success']:.0%}" + "".join(f", {k} {s[k]:.0%}" for k in ("stop", "continue") if k in s),
                  flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"written: {out}")


if __name__ == "__main__":
    main()

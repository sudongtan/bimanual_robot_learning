"""Residual RL on a frozen imitation policy (optional step 6 of the minimal plan, parked; docs/1tasks_scene.md).

The base policy (step 3's ACT with the one-hot input) plans a chunk of actions from the three cameras every
PLAN_EVERY frames, as LeRobot runs ACT without temporal ensembling but replanning sooner. A residual adds a bounded
correction (at most RES_SCALE per joint) to every frame's command, from the joints and the simulator's state (cube,
drawer), which the camera policy doesn't get. This follows ResiP (Ankile et al. 2024: a frozen chunked imitation
policy plus a closed-loop residual trained with PPO) and Policy Decorator (Yuan et al. 2025: a bounded residual on
a frozen Diffusion Policy or Behavior Transformer). Images are rendered only when the base plans, so an episode costs
~1/PLAN_EVERY of an evaluation rollout's rendering, which is ~95% of its time.

The reward is the episode's result, scored by sim/evaluate.py's Tracker exactly as an evaluation: success, and for
S1 (and S3) the stop as well (success and holding still at the end). The residual is trained on training seeds;
EVAL_SEEDS stay held out for evaluation.

    python -m sim.residual CKPT --variants S1:A --episodes 10       # step 5a: zero residual (the base alone), timed
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
from sim import evaluate as V
from sim import randomize as RZ
from sim import record as R
from sim.scene import CONTROL_EVERY

PLAN_EVERY = 25  # frames between base plans (0.5 s at 50 fps); ResiP's base plans ~1 s ahead, its residual acts every step
RES_SCALE = 0.05  # rad, the largest correction per joint and frame (Policy Decorator bounds its residual the same way)


def feature_size() -> int:
    """Length of `features`: 12 joint positions, 12 velocities, 12 base commands, cube position (3), cube offset in
    the drawer from its start (2), drawer opening, cube uprightness, time (3 together), and the one-hot variant (11)."""
    return 12 * 3 + 3 + 2 + 3 + len(RZ.VARIANT_DIMS)


def features(sim, qadr, dof, start, base, vector, t: int, horizon: int) -> np.ndarray:
    """The residual's observation: what the camera policy sees as joints, plus the simulator's state."""
    return np.concatenate([sim.data.qpos[qadr], sim.data.qvel[dof], base, sim.body_pos("cube"),
                           E.cube_in_drawer_frame(sim) - start["cube_drawer_xy"],
                           [sim.drawer_opening(), sim.data.body("cube").xmat[8], t / horizon], vector]).astype(np.float32)


def episode(policy, pre, post, variant, seed: int, rand: RZ.Randomization, residual=None,
            plan_every: int = PLAN_EVERY, timing: dict | None = None) -> dict:
    """One episode of base + residual. `residual(features) -> correction` in [-1, 1] per joint, scaled by
    RES_SCALE (None: no correction, the base alone). Returns the evaluation's result dict plus `reward`.
    `timing`, if given, accumulates seconds spent rendering, in the base policy, in physics and in the residual."""
    t0 = time.perf_counter()
    sim, start = E.reset(variant, seed, dataclasses.replace(rand, demo_noise=0.0))
    sim.control_every = 1  # ctrl is set once per frame and held for CONTROL_EVERY steps below
    qadr = R.joint_qpos(sim)
    dof = np.array([sim.model.jnt_dofadr[sim.model.actuator_trnid[i, 0]] for i in range(sim.model.nu)])
    renderers = {key: mujoco.Renderer(sim.model, *policy.image_size) for key in R.CAMERAS}
    vector = RZ.variant_vector(*variant)
    tracker = V.Tracker(sim, variant)
    horizon = V.HORIZON[variant[0]]
    clock = {"render": 0.0, "base": 0.0, "physics": 0.0, "residual": 0.0}
    clock["physics"] += time.perf_counter() - t0
    for t in range(horizon):
        if t % plan_every == 0:
            t1 = time.perf_counter()
            obs = V.observation(sim, qadr, renderers, start["command"], vector if policy.variant_input else None)
            t2 = time.perf_counter()
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(obs)))[0].cpu().numpy()
            clock["render"] += t2 - t1
            clock["base"] += time.perf_counter() - t2
        base = chunk[t % plan_every]
        ctrl = base
        if residual is not None:
            t1 = time.perf_counter()
            ctrl = base + RES_SCALE * np.clip(residual(features(sim, qadr, dof, start, base, vector, t, horizon)), -1, 1)
            clock["residual"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        sim.data.ctrl[:] = ctrl
        sim.step(CONTROL_EVERY)
        tracker.update(sim)
        clock["physics"] += time.perf_counter() - t1
    for r in renderers.values():
        r.close()
    result = tracker.result(sim, start, seed)
    result["reward"] = float(result.get("stopped", result["success"]))
    if timing is not None:
        for k, v in clock.items():
            timing[k] = timing.get(k, 0.0) + v
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", type=Path, help="the base policy's pretrained_model directory (ACT)")
    p.add_argument("--variants", default="S1:A", help="as sim.record, e.g. S1:A or T1:B:left")
    p.add_argument("--episodes", type=int, default=10, help="per variant, on the first EVAL_SEEDS (as sim.evaluate)")
    p.add_argument("--plan-every", type=int, default=PLAN_EVERY)
    p.add_argument("--out", type=Path, help="write a summary JSON here")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    policy, pre, post = V.load_policy(a.checkpoint, a.device)
    summary = {"checkpoint": str(a.checkpoint), "plan_every": a.plan_every, "residual": None, "variants": {}}
    for variant in R.parse_variants(a.variants):
        name = ":".join(x for x in variant if x)
        timing, results = {}, []
        for seed in RZ.EVAL_SEEDS[: a.episodes]:
            t = time.perf_counter()
            r = episode(policy, pre, post, variant, seed, RZ.Randomization(), plan_every=a.plan_every, timing=timing)
            results.append(r)
            failed = [c for c, v in r["checks"].items() if not v]
            print(f"{name} seed {seed}: reward {r['reward']:.0f} {'success' if r['success'] else 'fail ' + ','.join(failed)}"
                  f", cube slid {r['cube_slide_mm']:.1f} mm ({time.perf_counter() - t:.1f} s)", flush=True)
        n = len(results)
        s = {"episodes": n, "reward": float(np.mean([r["reward"] for r in results])),
             "success": float(np.mean([r["success"] for r in results])),
             "seconds_per_episode": {k: round(v / n, 2) for k, v in timing.items()}}
        summary["variants"][name] = s
        print(f"{name}: reward {s['reward']:.0%}, success {s['success']:.0%}, s/episode {s['seconds_per_episode']}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

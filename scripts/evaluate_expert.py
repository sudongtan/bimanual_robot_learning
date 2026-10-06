"""Evaluate a task's scripted expert on the evaluation seeds.

Normally run through scripts/evaluate.sh (`scripts/evaluate.sh <task> expert`), which
evaluates every kind of policy the same way. Trained policies go through LeRobot's
`lerobot-eval`; the scripted expert cannot, because `lerobot-eval` only loads LeRobot
policy checkpoints. So this script:

  1. wraps the expert behind LeRobot's policy interface (`reset`, `select_action`):
     it plans on the first step of an episode from the true simulator state, then
     plays the plan back;
  2. builds the task's environment through our LeRobot registration (`waste_sort`,
     sim/lerobot_plugin), without cameras — the expert does not look at images;
  3. runs one episode per seed with LeRobot's own episode runner (`rollout()`, the one
     `lerobot-eval` uses), counts successes as LeRobot's `eval_policy()` does, and
     writes `eval_info.json` (+ videos of the first episodes).

Usage:
    uv run python scripts/evaluate_expert.py --task t0_can_to_metal_bin --seed 1000 --n-episodes 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim.lerobot_plugin  # noqa: E402,F401  (registers the waste_sort env with LeRobot)
from sim.experts import make_expert  # noqa: E402
from sim.lerobot_plugin import WasteSortEnvConfig  # noqa: E402
from sim.waste_env import TASKS  # noqa: E402


class ExpertPolicy(nn.Module):
    """The scripted expert behind LeRobot's policy interface (`reset`, `select_action`).
    It reads the simulator state, not the observation: it plans on the first step of each
    episode, after the environment has been reset, and then follows its plan."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.expert = make_expert(env)
        self.k = 0

    def reset(self):
        self.k = 0

    def select_action(self, observation):
        if self.k == 0:
            self.expert.reset()
        a = self.expert.act(self.k)
        self.k += 1
        return torch.as_tensor(np.asarray(a, np.float32)).unsqueeze(0)


def identity_processors():
    """Pre/post-processors that change nothing — with the converters LeRobot uses for
    actions (an identity pipeline without them rejects a tensor, debug-log #27)."""
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    return (PolicyProcessorPipeline(steps=[]),
            PolicyProcessorPipeline(steps=[], to_transition=policy_action_to_transition,
                                    to_output=transition_to_policy_action))


def make_vec_env(env_cfg):
    """The single-environment vector env LeRobot's evaluator would build for `env_cfg`."""
    from lerobot.envs.factory import make_env

    envs = make_env(env_cfg, n_envs=1)
    return next(iter(next(iter(envs.values())).values()))


def evaluate(vec, env_cfg, policy, preprocessor, postprocessor, first_seed: int, n_episodes: int,
             out: Path, label: str, n_videos: int = 2) -> dict:
    """Run `n_episodes` from `first_seed` and write `<out>/eval_info.json` (+ videos)."""
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.io_utils import write_video

    out.mkdir(parents=True, exist_ok=True)
    env_pre, env_post = env_cfg.get_env_processors()
    per_episode, t0 = [], time.time()
    for i in range(n_episodes):
        seed = first_seed + i
        frames = []

        def render(env, frames=frames):
            frames.append(env.envs[0].render())

        data = rollout(env=vec, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                       preprocessor=preprocessor, postprocessor=postprocessor, seeds=[seed],
                       render_callback=render if i < n_videos else None)
        # as in eval_policy(): data up to the first done step counts
        done_ix = int(torch.argmax(data["done"][0].to(int)))
        success = bool(data["success"][0, : done_ix + 1].any())
        ep = {"episode_ix": i, "seed": seed, "success": success,
              "sum_reward": float(data["reward"][0, : done_ix + 1].sum()),
              "steps": done_ix + 1, "time_s": (done_ix + 1) / env_cfg.fps}
        if frames:
            path = out / "videos" / f"eval_episode_{i}.mp4"
            path.parent.mkdir(exist_ok=True)
            write_video(str(path), np.stack(frames[: done_ix + 2]), env_cfg.fps)
            ep["video"] = str(path)
        per_episode.append(ep)
        print(f"seed {seed}: {'success' if success else 'FAIL   '} after {ep['time_s']:.1f} s")

    succ = [e["success"] for e in per_episode]
    info = {
        "task": env_cfg.task, "policy": label, "seeds": [first_seed, first_seed + n_episodes - 1],
        "per_episode": per_episode,
        "aggregated": {
            "pc_success": 100 * float(np.mean(succ)),
            "avg_sum_reward": float(np.mean([e["sum_reward"] for e in per_episode])),
            "avg_time_to_success_s": float(np.mean([e["time_s"] for e in per_episode if e["success"]]))
            if any(succ) else None,
            "eval_s": time.time() - t0,
        },
    }
    (out / "eval_info.json").write_text(json.dumps(info, indent=2))
    print(f"{sum(succ)}/{len(succ)} successful ({info['aggregated']['pc_success']:.0f}%) -> {out / 'eval_info.json'}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t0_can_to_metal_bin", choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, required=True, help="first seed; episodes use seed, seed+1, ...")
    ap.add_argument("--n-episodes", type=int, required=True)
    ap.add_argument("--videos", type=int, default=2, help="save an overview video of the first N episodes")
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/eval/<task>/expert")
    args = ap.parse_args()

    cfg = WasteSortEnvConfig(task=args.task, cameras=[])  # the expert does not look at images
    vec = make_vec_env(cfg)
    policy = ExpertPolicy(vec.envs[0].unwrapped)
    pre, post = identity_processors()  # the expert's actions need no (de)normalisation
    evaluate(vec, cfg, policy, pre, post, args.seed, args.n_episodes,
             args.out or Path("outputs/eval") / args.task / "expert", "scripted expert", args.videos)
    vec.close()


if __name__ == "__main__":
    main()

"""Early checks on a training checkpoint, before spending rollouts on it (docs/3train_evaluation.md, findings 1 and 8).

    python -m sim.offline_check CKPT data/min_onehot              # stats + replay of 6 training episodes, on the CPU
    python -m sim.offline_check CKPT data/min --episodes 12

Two checks, both without the simulator, so they run beside a GPU job:

    stats    the checkpoint's normalisation stats keep the floor (dataset_view.STD_FLOOR, RANGE_FLOOR) on every state
             and action dimension. Finding 1: without it a never-moving joint turns physics jitter into inputs of 15-50.
    replay   the policy runs through recorded training episodes frame by frame, as in a rollout (select_action, its
             own chunking and observation history), but always sees the recorded observation, so errors don't
             compound. A healthy checkpoint reproduces the recorded actions closely (ACT at 40k: 0.003-0.009 rad
             mean per joint); a much larger error points at the inputs (features, normalisation, image size) rather
             than the training length.

Any value past its threshold is listed under FLAGS at the end.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from sim import evaluate as V
from sim import record as R
from sim.dataset_view import RANGE_FLOOR, STD_FLOOR

REPLAY_FLAG = 0.03  # rad, mean error per joint; ~3-10x what a working ACT checkpoint reaches


def check_stats(path: Path) -> list[str]:
    from safetensors.torch import load_file

    flags = []
    stats = load_file(glob.glob(str(path / "policy_preprocessor_step_*_normalizer_processor.safetensors"))[0])
    for key in ("observation.state", "action"):
        std, lo, hi = (stats[f"{key}.{s}"].numpy() for s in ("std", "min", "max"))
        print(f"stats {key}: smallest std {std.min():.4f}, smallest range {(hi - lo).min():.4f}")
        if std.min() < STD_FLOOR * 0.999:
            flags.append(f"{key}: std below the floor on dims {np.flatnonzero(std < STD_FLOOR * 0.999).tolist()}")
        if (hi - lo).min() < RANGE_FLOOR * 0.999:
            flags.append(f"{key}: min-max range below the floor on dims {np.flatnonzero(hi - lo < RANGE_FLOOR * 0.999).tolist()}")
    return flags


def replay(policy, pre, post, ds, index: int) -> np.ndarray:
    """Absolute error per frame and joint between the policy's action and the recorded one."""
    ep = ds.meta.episodes[index]
    policy.reset()
    errors = []
    for i in range(ep["dataset_from_index"], ep["dataset_to_index"]):
        x = ds[i]
        # The dataset's own names: SmolVLA's saved preprocessor renames the cameras to its own (camera1, ...)
        obs = {k: x[k][None] for k in x if k.startswith("observation.")}
        obs["task"] = [x["task"]]
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))[0].cpu().numpy()
        errors.append(np.abs(action - x["action"].numpy()))
    return np.array(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", type=Path, help="a pretrained_model directory")
    p.add_argument("root", type=Path, help="the dataset it was trained on")
    p.add_argument("--episodes", type=int, default=6, help="spread evenly over the dataset")
    p.add_argument("--device", default="cpu", help="cpu by default, to leave the GPU to a running job")
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    flags = check_stats(a.checkpoint)
    policy, pre, post = V.load_policy(a.checkpoint, a.device)
    ds = LeRobotDataset(R.REPO_ID, root=a.root)
    n = ds.meta.total_episodes
    for index in np.linspace(0, n - 1, a.episodes).round().astype(int):
        e = replay(policy, pre, post, ds, int(index))
        task = ds.meta.episodes[int(index)]["tasks"][0]
        print(f"episode {index} ({len(e)} frames, '{task}'): mean error {e.mean():.4f} rad, "
              f"worst joint {e.mean(0).max():.4f}, largest {e.max():.3f}", flush=True)
        if e.mean(0).max() > REPLAY_FLAG:
            flags.append(f"episode {index}: joint {e.mean(0).argmax()} mean error {e.mean(0).max():.3f} rad")
    print("FLAGS:" if flags else "FLAGS: none")
    for f in flags:
        print("  " + f)


if __name__ == "__main__":
    main()

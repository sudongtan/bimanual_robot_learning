"""Phase 2, part 1: run the scripted expert and save states and actions — no rendering.

One file per episode, data/raw/<task>/seed_<seed>.npz, holding everything needed to
re-render any camera at any size later (scene-decisions D20):

    qpos         (T, nq)  full simulator state at each control step, before the action
    state        (T, 12)  the arm joint positions = the policy's state input
    action       (T, 12)  the joint targets sent at that step
    items        the items attached to the scene (to rebuild the same model)
    task, instruction, seed, success, steps_to_success

Only successful episodes are meant for training; failures are saved too (flagged)
so they can be inspected.

Usage:
    uv run python scripts/record_states.py --seeds $(seq 0 49)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.experts import make_expert  # noqa: E402
from sim.waste_env import CONTROL_HZ, TASKS, WasteSortEnv  # noqa: E402

SETTLE_AFTER_SUCCESS_S = 0.5  # keep recording briefly after success so the end state is in the data


def record(env: WasteSortEnv, expert, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    plan = expert.reset()
    qpos, state, action = [], [], []
    n_steps = min(env.max_steps, int((plan.duration + 1.5) * CONTROL_HZ))
    success_at = None
    for k in range(n_steps):
        a = expert.act(k).astype(np.float32)
        qpos.append(env.data.qpos.copy())
        state.append(obs["state"])
        action.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        if info["success"] and success_at is None:
            success_at = k + 1
        if success_at is not None and k + 1 - success_at >= SETTLE_AFTER_SUCCESS_S * CONTROL_HZ:
            break
        if truncated:
            break
    return {"qpos": np.array(qpos), "state": np.array(state, np.float32), "action": np.array(action, np.float32),
            "items": np.array(env.items), "task": env.task_name, "instruction": env.task.instruction,
            "seed": seed, "success": success_at is not None, "steps_to_success": success_at or -1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="can_to_metal_bin", choices=sorted(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    out = args.out / args.task
    out.mkdir(parents=True, exist_ok=True)
    env = WasteSortEnv(task=args.task)
    expert = make_expert(env)
    t0, ok = time.perf_counter(), 0
    for seed in args.seeds:
        ep = record(env, expert, seed)
        np.savez_compressed(out / f"seed_{seed}.npz", **ep)
        ok += ep["success"]
        print(f"seed {seed:4d}: {'success' if ep['success'] else 'FAIL   '} {len(ep['action']):4d} steps")
    print(f"{ok}/{len(args.seeds)} successful, {time.perf_counter() - t0:.1f} s total -> {out}")


if __name__ == "__main__":
    main()

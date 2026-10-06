"""Open one episode's scene (with its items placed) in the MuJoCo viewer.

The scene file holds no items (design-decisions D17); this builds the model the
environment uses for a task, resets it with a seed, and opens the viewer on it.

Usage:
    uv run python scripts/view_episode.py --task t0_can_to_metal_bin --seed 7     # any task in TASKS
    uv run mjpython scripts/view_episode.py --task t0_can_to_metal_bin --expert   # play the task's scripted expert
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.waste_env import CONTROL_HZ, TASKS, WasteSortEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t0_can_to_metal_bin", choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert", action="store_true", help="play the scripted expert (needs `mjpython` on macOS)")
    args = ap.parse_args()

    env = WasteSortEnv(task=args.task)
    env.reset(seed=args.seed)
    print(f"{args.task}, seed {args.seed}: {env.task.instruction}")
    if not args.expert:
        # Managed viewer: you drive it (Space to run, Load key resets the arms but not the items).
        mujoco.viewer.launch(env.model, env.data)
        return

    from sim.experts import make_expert
    expert = make_expert(env)
    plan = expert.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for k in range(int((plan.duration + 2.0) * CONTROL_HZ)):
            t0 = time.perf_counter()
            env.step(expert.act(k))
            viewer.sync()
            if not viewer.is_running():
                break
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()

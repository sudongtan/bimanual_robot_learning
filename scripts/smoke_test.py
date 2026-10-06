"""Run a task's scripted expert on a few seeds: success rate, per-phase log, video.

Task-agnostic: the expert comes from sim.experts.EXPERTS for --task.

Usage:
    uv run python scripts/smoke_test.py                                   # can_to_metal_bin, seeds 0-4
    uv run python scripts/smoke_test.py --task can_to_metal_bin --seeds 0 1 2 --video-all

Videos (overview camera, 25 fps) go to outputs/smoke/<task>/seed_<n>.mp4.
"""

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.experts import make_expert  # noqa: E402
from sim.waste_env import CONTROL_HZ, TASKS, WasteSortEnv  # noqa: E402

OUT = Path("outputs/smoke")
BINS = ("plastic", "metal")


def run(env: WasteSortEnv, expert, seed: int, video: bool, log: bool) -> dict:
    env.reset(seed=seed)
    start = {i: env.item_pos(i) for i in env.items}
    plan = expert.reset()
    frames, last_phase = [], None
    disturbed = None  # did the grasped item move >1 cm before the grasp? (knocked, not grasped)
    # Run the plan, then give items 1.5 s to land; stop early on success.
    n_steps = min(env.max_steps, int((plan.duration + 1.5) * CONTROL_HZ))
    info = {"success": False}
    for k in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(expert.act(k))
        phase = plan.phase(k / CONTROL_HZ)
        if log and phase != last_phase:
            items = "  ".join(f"{i} z {env.item_pos(i)[2]:.3f}" for i in env.items)
            bins = "  ".join(f"{b} bin {env.bin_opening(b):.3f}" for b in BINS)
            print(f"    t={k / CONTROL_HZ:5.2f}s  {phase:45s} {items}  {bins}")
            last_phase = phase
        if disturbed is None and phase.startswith(expert.grasp_phase):
            item = expert.grasped_item
            disturbed = bool(np.linalg.norm(env.item_pos(item)[:2] - start[item][:2]) > 0.01)
        if video and k % 2 == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    if video:
        (OUT / env.task_name).mkdir(parents=True, exist_ok=True)
        imageio.mimsave(OUT / env.task_name / f"seed_{seed}.mp4", frames, fps=CONTROL_HZ // 2)
    return {"seed": seed, "success": info["success"], "disturbed": bool(disturbed), "time": info.get("time", 0.0),
            "items": {i: (start[i].round(3), env.item_pos(i).round(3)) for i in env.items},
            "bins": {b: round(env.bin_opening(b), 3) for b in BINS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="can_to_metal_bin", choices=sorted(TASKS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--video-all", action="store_true", help="save a video for every seed, not just the first")
    ap.add_argument("--quiet", action="store_true", help="no per-phase log")
    args = ap.parse_args()

    env = WasteSortEnv(task=args.task)
    expert = make_expert(env)
    print(f"{args.task}: {env.task.instruction}")
    results = []
    for i, seed in enumerate(args.seeds):
        print(f"seed {seed}:")
        r = run(env, expert, seed, video=args.video_all or i == 0, log=not args.quiet)
        items = " | ".join(f"{name} {a} -> {b}" for name, (a, b) in r["items"].items())
        bins = ", ".join(f"{b} bin out {v} m" for b, v in r["bins"].items())
        flag = f" ({expert.grasped_item} moved >1 cm before the grasp)" if r["disturbed"] else ""
        print(f"  -> {'SUCCESS' if r['success'] else 'FAIL'}{flag} at t={r['time']:.1f}s | {items} | {bins}")
        results.append(r)
    n = sum(r["success"] for r in results)
    print(f"\n{n}/{len(results)} succeeded ({100 * n / len(results):.0f}%)")
    env.close()


if __name__ == "__main__":
    main()

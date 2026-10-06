"""Phase 2, part 2: render camera images from saved states into a LeRobot dataset.

Reads data/raw/<task>/seed_*.npz (from record_states.py), rebuilds each episode's
model, sets every saved state, renders the cameras, and writes a LeRobotDataset
(scene-decisions D20). No physics is re-run, so images match the recorded
states exactly. Rendering runs in parallel worker processes; the main process
writes the dataset (encoding videos) in episode order.

Usage:
    uv run python scripts/render_dataset.py                          # all 3 cameras, 224x224
    uv run python scripts/render_dataset.py --cameras overview left_wrist --size 96 96 --workers 6
    uv run python scripts/render_dataset.py --no-reflections         # scene-decisions D21 (open)

Output: data/lerobot/<task>/ (LeRobot v3 format, fps = control rate).
"""

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARM_JOINTS = [f"{s}_{j}" for s in ("left", "right")
              for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]

_worker = {}


def _init_worker(task: str, cameras: tuple[str, ...], size: tuple[int, int], reflections: bool) -> None:
    import mujoco
    from sim.waste_env import WasteSortEnv
    _worker.update(env=WasteSortEnv(task=task), cameras=cameras, size=size, reflections=reflections, mujoco=mujoco)


def _render_episode(path: str) -> tuple[str, dict]:
    """Render every saved state of one episode; returns {camera: (T, H, W, 3) uint8}."""
    mujoco, env = _worker["mujoco"], _worker["env"]
    ep = np.load(path)
    env._use_items(tuple(str(i) for i in ep["items"]))         # same model as when it was recorded
    m, d = env.model, env.data
    assert m.nq == ep["qpos"].shape[1], f"{path}: state size {ep['qpos'].shape[1]} vs model {m.nq}"
    r = mujoco.Renderer(m, *_worker["size"])
    frames = {c: np.empty((len(ep["qpos"]), *_worker["size"], 3), np.uint8) for c in _worker["cameras"]}
    for t, q in enumerate(ep["qpos"]):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for c in _worker["cameras"]:
            r.update_scene(d, camera=c)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = _worker["reflections"]
            frames[c][t] = r.render()
    r.close()
    return path, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="can_to_metal_bin")  # any task in sim.waste_env.TASKS with recorded episodes
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/lerobot"))
    ap.add_argument("--cameras", nargs="+", default=["overview", "left_wrist", "right_wrist"])
    ap.add_argument("--size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--no-reflections", action="store_true")
    ap.add_argument("--include-failures", action="store_true", help="also write unsuccessful episodes")
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (for a quick check)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from sim.waste_env import CONTROL_HZ

    files = sorted((args.raw / args.task).glob("seed_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    files = [f for f in files if args.include_failures or bool(np.load(f)["success"])][: args.limit]
    if not files:
        sys.exit(f"no episodes in {args.raw / args.task}")
    root = args.out / args.task
    if root.exists():
        shutil.rmtree(root)
    h, w = args.size
    features = {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": ARM_JOINTS},
        "action": {"dtype": "float32", "shape": (12,), "names": ARM_JOINTS},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
           for c in args.cameras},
    }
    ds = LeRobotDataset.create(repo_id=f"local/{args.task}", fps=CONTROL_HZ, features=features, root=root,
                               robot_type="so101_bimanual_sim", use_videos=True)

    t0, n_frames = time.perf_counter(), 0
    ctx = mp.get_context("spawn")
    init = (args.task, tuple(args.cameras), (h, w), not args.no_reflections)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
        for i, (path, frames) in enumerate(pool.imap(_render_episode, [str(f) for f in files])):
            ep = np.load(path)
            instruction = str(ep["instruction"])
            for t in range(len(ep["action"])):
                ds.add_frame({"observation.state": ep["state"][t], "action": ep["action"][t], "task": instruction,
                              **{f"observation.images.{c}": frames[c][t] for c in args.cameras}})
            ds.save_episode()
            n_frames += len(ep["action"])
            print(f"episode {i:3d} ({Path(path).stem}): {len(ep['action'])} frames, "
                  f"{time.perf_counter() - t0:6.1f} s elapsed")
    ds.finalize()
    dt = time.perf_counter() - t0
    print(f"{len(files)} episodes, {n_frames} frames in {dt:.0f} s ({n_frames / dt:.0f} frames/s) -> {root}")


if __name__ == "__main__":
    main()

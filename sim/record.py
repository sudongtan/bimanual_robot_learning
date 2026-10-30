"""Record scripted episodes as one LeRobot dataset (design: 2data_collection.md).

    python -m sim.record data/pilot --per-variant 5                  # step 0: 5 of each of the 20 variants
    python -m sim.record data/s1a --variants S1:A --per-variant 50   # plan step 1
    python -m sim.record data/t1 --variants T1:A:middle,T1:B:left    # task:arm[:target]; arm = opener (T1) / placer (T2)
    python -m sim.record data/dbg --variants S1:A --per-variant 2 --factors none   # no randomisation

Each variant walks its own training seeds (`train_seeds`, sim/randomize.py) in order until it has
`--per-variant` successful episodes (S2: half from each start kind). The expert runs in worker
processes and returns, per 50 fps frame, the physics state before the frame and the command held
during it. The main process then rebuilds the episode's model from its seed, replays the commands
from the first state and checks that every recorded state comes back exactly (first-batch check 1),
renders the three cameras from the recorded states and writes the frames.

Written under `root`:
    LeRobot dataset    successful episodes only, so it trains as is. Per frame: observation.state (12 actuated
                       joints), action (the command sent, with demonstration noise), action_nominal (without),
                       observation.images.{overview,left_wrist,right_wrist} (HEIGHT x WIDTH, 240 x 320 by default), subtask_index
                       (composites: 0, then 1 once the first arm is home), variant_index (into VARIANTS, for
                       the one-hot input); the episode's task is its command.
    sim/episodes.jsonl every episode run, failed ones too: variant, seed, the run_* result (success, command,
                       scene, checks) and episode_index in the dataset (None if failed).
    sim/states/*.npz   per episode run, float64: state (STATE_SPEC, before each frame), action, action_nominal,
                       subtask. Enough to replay exactly or re-render any camera.
    sim/info.json      git commit, control rate, randomisation settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from sim import randomize as RZ
from sim import skills as K
from sim.dataset_view import STD_FLOOR, floor_std
from sim.scene import CONTROL_EVERY, STATE_SPEC
from sim.validate import config

FPS = 50
HEIGHT, WIDTH = 240, 320  # half ACT/ALOHA's 480 x 640, for training speed (2data_collection.md, design point 4); --size
CAMERAS = {"overview": "overview", "left_wrist": "left/wrist_cam", "right_wrist": "right/wrist_cam"}
REPO_ID = "local/bimanual_so101"
ROBOT_TYPE = "bimanual_so101_sim"


class Frames:
    """`rec` for a run_*: at each control tick, the state before it, the command held during it and the subtask."""

    def __init__(self):
        self.states, self.actions, self.nominal, self.subtask = [], [], [], []
        self.held = True  # every command stayed the same until the next tick
        self.sim = None

    def __call__(self, s):
        self.sim = s
        if s.ticked:
            self.states.append(s.tick_state.copy())
            self.actions.append(s.applied_ctrl.copy())
            self.nominal.append(s.nominal_ctrl.copy())
            self.subtask.append(s.subtask)
        elif self.actions:
            self.held &= bool(np.array_equal(s.applied_ctrl, self.actions[-1]))

    def arrays(self) -> dict:
        return dict(state=np.array(self.states), action=np.array(self.actions),
                    action_nominal=np.array(self.nominal), subtask=np.array(self.subtask, dtype=np.int64))


def run(variant, seed: int, rec=None, rand: RZ.Randomization = RZ.Randomization()) -> dict:
    """Play one variant (task, arm, target) with `seed`."""
    task, arm, target = variant
    args = (arm,) if target is None else (arm, target)
    return getattr(K, f"run_{task}")(*args, seed=seed, rec=rec, rand=rand)


def _play(job) -> tuple[dict, dict]:
    variant, seed, rand = job
    f = Frames()
    result = run(variant, seed, f, rand)
    assert f.held, (variant, seed)
    return result, f.arrays()


def wanted(variant, n: int) -> dict:
    """Successful episodes needed per start kind: S2 half and half (cube on the table / in the drawer), else all."""
    if variant[0] == "S2":
        return {False: n - n // 2, True: n // 2}
    return {None: n}


def collect(variant, n: int, rand: RZ.Randomization, pool: ProcessPoolExecutor, batch: int):
    """Yield (result, frames) in seed order through the variant's training seeds until it has `n`
    successful episodes. Seeds run in parallel batches; results past the last needed one are dropped,
    so what is recorded doesn't depend on the number of workers. Episodes of an S2 start kind that is
    already full are skipped."""
    need = wanted(variant, n)
    seeds = iter(RZ.train_seeds(*variant))
    while any(need.values()):
        jobs = [(variant, s, rand) for _, s in zip(range(batch), seeds)]
        if not jobs:
            raise RuntimeError(f"{variant}: out of training seeds, still need {need}")
        for result, frames in pool.map(_play, jobs):
            kind = result.get("cube_in_drawer")
            if not need.get(kind):
                continue
            if result["success"]:
                need[kind] -= 1
            yield result, frames
            if not any(need.values()):
                return


def replay_matches(model, frames: dict) -> bool:
    """Replay the commands from the first recorded state, each held CONTROL_EVERY steps: does every
    recorded state come back exactly?"""
    data = mujoco.MjData(model)
    states = frames["state"]
    mujoco.mj_setState(model, data, states[0], STATE_SPEC)
    state = np.empty_like(states[0])
    for k, action in enumerate(frames["action"][:-1]):
        data.ctrl[:] = action
        mujoco.mj_step(model, data, CONTROL_EVERY)
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        if not np.array_equal(state, states[k + 1]):
            return False
    return True


def features(sim, images: bool, size=(HEIGHT, WIDTH)) -> dict:
    joints = [sim.model.actuator(i).name for i in range(sim.model.nu)]
    vec = lambda dtype, names: {"dtype": dtype, "shape": (len(names),), "names": names}
    f = {"observation.state": vec("float32", joints), "action": vec("float32", joints),
         "action_nominal": vec("float32", joints),
         "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
         "variant_index": {"dtype": "int64", "shape": (1,), "names": None}}
    for key in CAMERAS if images else ():
        f[f"observation.images.{key}"] = {"dtype": "video", "shape": (*size, 3),
                                          "names": ["height", "width", "channels"]}
    return f


def joint_qpos(sim) -> np.ndarray:
    """qpos addresses of the 12 actuated joints, in actuator order (each arm's 5 joints, then its gripper)."""
    return np.array([sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]] for i in range(sim.model.nu)])


def write_episode(ds, variant, result: dict, frames: dict, rand: RZ.Randomization, images: bool,
                  size=(HEIGHT, WIDTH)):
    """Rebuild the episode's model (colours and lighting come from the seed), check the replay, then
    render each frame from its recorded state and add it to the dataset."""
    sim, _ = K.new_sim(result["seed"], rand)
    model = sim.model
    if not replay_matches(model, frames):
        raise RuntimeError(f"replay does not match the recording: {variant} seed {result['seed']}")
    qadr = joint_qpos(sim)
    vis = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *size) if images else None
    v = RZ.VARIANTS.index(variant)
    for k, state in enumerate(frames["state"]):
        mujoco.mj_setState(model, vis, state, STATE_SPEC)
        mujoco.mj_forward(model, vis)
        frame = {"observation.state": vis.qpos[qadr].astype(np.float32),
                 "action": frames["action"][k].astype(np.float32),
                 "action_nominal": frames["action_nominal"][k].astype(np.float32),
                 "subtask_index": np.array([frames["subtask"][k]]),
                 "variant_index": np.array([v]),
                 "task": result["command"]}
        if images:
            for key, cam in CAMERAS.items():
                renderer.update_scene(vis, camera=cam)
                frame[f"observation.images.{key}"] = renderer.render()
        ds.add_frame(frame)
    ds.save_episode()
    if renderer is not None:
        renderer.close()


def to_json(obj) -> str:
    """One log line; numpy scalars and arrays (e.g. a bool from a numpy comparison) as plain JSON."""
    return json.dumps(obj, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())


def git_commit() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}  # cwd is sim/


def parse_variants(spec: str) -> list[tuple]:
    if spec == "all":
        return list(RZ.VARIANTS)
    out = []
    for item in spec.split(","):
        task, arm, *target = item.split(":")
        variant = (task, arm, target[0] if target else None)
        if variant not in RZ.VARIANTS:
            raise ValueError(f"unknown variant {item!r}; e.g. S1:A, S4:B:left, T1:A:middle")
        out.append(variant)
    return out


def _write_shard(job) -> int:
    """Write consecutive successful episodes (each loaded from its states file) as one small dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shard, entries, states_dir, rand, images, size = job
    sim, _ = K.new_sim(0, rand)
    ds = LeRobotDataset.create(f"local/{shard.name}", fps=FPS, features=features(sim, images, size), root=shard,
                               robot_type=ROBOT_TYPE, use_videos=True, image_writer_threads=len(CAMERAS) if images else 0)
    for e in entries:
        with np.load(states_dir / e["states"]) as f:
            frames = dict(f)
        write_episode(ds, tuple(e["variant"]), e, frames, rand, images, size)
    ds.finalize()
    return len(entries)


def record(root: Path, variants: list[tuple], n: int, rand: RZ.Randomization, images: bool = True, workers=None,
           size=(HEIGHT, WIDTH)):
    """Two passes. 1: the expert walks each variant's seeds (in parallel) and every episode's frames are
    saved. 2: the successful episodes, in order, are split into one consecutive chunk per worker; each
    worker checks the replay, renders and writes a small dataset (rendering is ~16 ms per camera image,
    ~95% of the time), and the chunks are merged in order (LeRobot's aggregate_datasets), so episode
    indices don't depend on the number of workers."""
    from lerobot.datasets.aggregate import aggregate_datasets

    if root.exists():
        raise FileExistsError(f"{root} exists; record into a new directory")
    tmp = root.parent / f".{root.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "states").mkdir(parents=True)
    workers = workers or os.cpu_count()
    entries = []
    with ProcessPoolExecutor(workers) as pool:
        for variant in variants:
            task, arm, target = variant
            ok = failed = 0
            for result, frames in collect(variant, n, rand, pool, workers):
                name = f"{task}_{arm}_{target or '-'}_{result['seed']:05d}"
                np.savez_compressed(tmp / "states" / f"{name}.npz", **frames)
                entries.append({"variant": list(variant), "episode_index": None, "frames": len(frames["state"]),
                                "states": f"{name}.npz", **result})
                ok, failed = ok + result["success"], failed + (not result["success"])
            print(f"{task} {arm} {target or ''}: {ok} episodes, {failed} failed", flush=True)

        good = [e for e in entries if e["success"]]
        for i, e in enumerate(good):
            e["episode_index"] = i
        chunks = [c.tolist() for c in np.array_split(np.array(good, dtype=object), min(workers, len(good)))]
        shards = [tmp / f"shard{i:03d}" for i in range(len(chunks))]
        jobs = [(shard, chunk, tmp / "states", rand, images, size) for shard, chunk in zip(shards, chunks)]
        print(f"rendering and writing {len(good)} episodes in {len(jobs)} parts", flush=True)
        list(pool.map(_write_shard, jobs))

    aggregate_datasets([f"local/{s.name}" for s in shards], REPO_ID, roots=shards, aggr_root=root)
    floor_std(root)
    side = root / "sim"
    side.mkdir()
    shutil.move(tmp / "states", side / "states")
    with open(side / "episodes.jsonl", "w") as log:
        for e in entries:
            log.write(to_json({**e, "states": f"states/{e['states']}"}) + "\n")
    (side / "info.json").write_text(json.dumps(
        {**git_commit(), "fps": FPS, "control_every": CONTROL_EVERY, "per_variant": n, "images": images, "image_size": list(size),
         "randomization": asdict(rand), "variants": [list(v) for v in variants], "std_floor": STD_FLOOR}, indent=1))
    shutil.rmtree(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path, help="new directory for the dataset")
    p.add_argument("--variants", default="all", help="'all' or comma-separated task:arm[:target]")
    p.add_argument("--per-variant", type=int, default=50, help="successful episodes per variant")
    p.add_argument("--factors", default="all", help="randomisation factors, as in sim.validate ('all', 'none', ...)")
    p.add_argument("--no-images", action="store_true", help="skip rendering (states can be re-rendered later)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--size", default=f"{HEIGHT}x{WIDTH}", help="camera image height x width, e.g. 480x640")
    a = p.parse_args()
    size = tuple(int(x) for x in a.size.split("x"))
    record(a.root, parse_variants(a.variants), a.per_variant, config(a.factors), not a.no_images, a.workers, size)


if __name__ == "__main__":
    main()

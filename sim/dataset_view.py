"""A training view of a recorded dataset: the clean action labels, the variant input, or both.

    python -m sim.dataset_view data/s1a data/s1a_nominal --nominal          # plan step 1
    python -m sim.dataset_view data/min data/min_onehot --variant-input     # minimal plan step 3
    python -m sim.dataset_view data/min data/min_view --nominal --variant-input

--nominal          LeRobot trains on the `action` column; swap `action` and `action_nominal` (frames,
                   dataset and per-episode statistics), so the same training command trains on the
                   expert's command without demonstration noise (DART's clean labels; 2data_collection.md,
                   design point 2).
--variant-input    add `observation.environment_state`, the one-hot variant vector (RZ.variant_vector,
                   from each frame's variant_index). LeRobot's ACT and DP take it as an extra encoder
                   input, unnormalised, as Meta-World MT10/MT50 adds a task ID. sim/evaluate.py supplies
                   it in rollouts when the policy has it.

Every view, and every recording (sim/record.py), also gets STD_FLOOR: see floor_std.

The videos and the recorder's own files (`sim/`) are linked, not copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sim import randomize as RZ

A, B = "action", "action_nominal"
ENV = "observation.environment_state"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
# rad; the smallest std the normaliser divides by. A joint that never moves in a dataset (the idle arm's
# gripper in S1: std 7e-9) otherwise turns physics jitter of ~3e-7 rad into normalised inputs of 15-50, and
# the network can tell episodes apart by it: low loss on recorded frames, failure in closed loop (plan step 1).
# The smallest real spread is 0.017 (start noise), so 0.01 changes nothing that varies. Diffusion Policy's
# normaliser likewise treats near-constant dimensions as constant instead of dividing by their range.
STD_FLOOR = 0.01
FLOORED = ("observation.state", A, B)


def floor_std(root: Path, floor: float = STD_FLOOR):
    """Raise the std of the joint features in `root`'s meta/stats.json (what LeRobot normalises with) to `floor`."""
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    for k in FLOORED:
        if k in stats:
            stats[k]["std"] = np.maximum(stats[k]["std"], floor).tolist()
    path.write_text(json.dumps(stats, indent=4))


def swap_name(name: str) -> str:
    """`action` <-> `action_nominal`, also inside `stats/<feature>/<stat>` column names."""
    for x, y in ((A, B), (B, A)):
        if name == x:
            return y
        if name.startswith(f"stats/{x}/"):
            return f"stats/{y}/" + name[len(f"stats/{x}/"):]
    return name


def stats(x: np.ndarray) -> dict:
    """LeRobot's per-feature statistics of an (n, d) array."""
    s = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    s.update({f"q{round(q * 100):02d}": np.quantile(x, q, axis=0) for q in QUANTILES})
    return {k: v.tolist() for k, v in s.items()}


def view(src: Path, dst: Path, nominal: bool, variant_input: bool):
    if dst.exists():
        raise FileExistsError(dst)
    rename = swap_name if nominal else (lambda c: c)
    vectors = np.stack([RZ.variant_vector(*v) for v in RZ.VARIANTS])
    env_all, env_by_episode = [], {}
    for f in sorted(src.glob("data/**/*.parquet")):
        t = pq.read_table(f)
        t = t.rename_columns([rename(c) for c in t.column_names])
        if variant_input:
            env = vectors[t["variant_index"].to_numpy()]
            t = t.append_column(ENV, pa.array(list(env), type=pa.list_(pa.float32())))
            env_all.append(env)
            for ep in np.unique(t["episode_index"].to_numpy()):
                env_by_episode[int(ep)] = env[t["episode_index"].to_numpy() == ep]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
    for f in sorted(src.glob("meta/episodes/**/*.parquet")):
        t = pq.read_table(f)
        t = t.rename_columns([rename(c) for c in t.column_names])
        if variant_input:
            per = [stats(env_by_episode[int(ep)]) for ep in t["episode_index"].to_numpy()]
            for k in per[0]:
                kind = pa.int64() if k == "count" else pa.float64()
                t = t.append_column(f"stats/{ENV}/{k}", pa.array([p[k] for p in per], type=pa.list_(kind)))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    meta = json.loads((src / "meta" / "stats.json").read_text())
    meta = {rename(k): v for k, v in meta.items()}
    info = json.loads((src / "meta" / "info.json").read_text())
    if variant_input:
        meta[ENV] = stats(np.concatenate(env_all))
        info["features"][ENV] = {"dtype": "float32", "shape": [len(RZ.VARIANT_DIMS)], "names": list(RZ.VARIANT_DIMS)}
    (dst / "meta" / "stats.json").write_text(json.dumps(meta, indent=4))
    floor_std(dst)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    for name in ("videos", "sim"):
        if (src / name).exists():
            (dst / name).symlink_to((src / name).resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--nominal", action="store_true", help="train on action_nominal")
    p.add_argument("--variant-input", action="store_true", help=f"add {ENV}, the one-hot variant vector")
    a = p.parse_args()
    if not (a.nominal or a.variant_input):
        p.error("nothing to change: pass --nominal and/or --variant-input")
    view(a.src, a.dst, a.nominal, a.variant_input)
    print(f"written: {a.dst}")


if __name__ == "__main__":
    main()

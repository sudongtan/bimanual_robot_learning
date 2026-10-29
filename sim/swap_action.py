"""A view of a recorded dataset that trains on `action_nominal` (2data_collection.md, design point 2).

    python -m sim.swap_action data/s1a data/s1a_nominal

LeRobot trains on the `action` column. This writes a copy whose `action` and `action_nominal` are
swapped, in the frames and in the dataset and per-episode statistics, so the same training command
trains on the expert's command without demonstration noise (DART's clean labels). The videos and the
recorder's own files (`sim/`) are linked, not copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

A, B = "action", "action_nominal"


def swap_name(name: str) -> str:
    """`action` <-> `action_nominal`, also inside `stats/<feature>/<stat>` column names."""
    for x, y in ((A, B), (B, A)):
        if name == x:
            return y
        if name.startswith(f"stats/{x}/"):
            return f"stats/{y}/" + name[len(f"stats/{x}/"):]
    return name


def swap(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(dst)
    for f in sorted(src.glob("data/**/*.parquet")) + sorted(src.glob("meta/episodes/**/*.parquet")):
        t = pq.read_table(f)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.rename_columns([swap_name(c) for c in t.column_names]), out)
    stats = json.loads((src / "meta" / "stats.json").read_text())
    (dst / "meta" / "stats.json").write_text(json.dumps({swap_name(k): v for k, v in stats.items()}, indent=4))
    for name in ("info.json", "tasks.parquet"):
        shutil.copy(src / "meta" / name, dst / "meta" / name)
    for name in ("videos", "sim"):
        if (src / name).exists():
            (dst / name).symlink_to((src / name).resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    a = p.parse_args()
    swap(a.src, a.dst)
    print(f"written: {a.dst} (action = {a.src}'s action_nominal)")


if __name__ == "__main__":
    main()

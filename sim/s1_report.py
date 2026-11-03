"""S1 reported apart from the slide limit (docs/3train_evaluation.md, finding 9), from logged rollouts.

    python -m sim.s1_report outputs/min_act/eval_100000_te/train outputs/min_smolvla/eval/train

S1's success needs every check, and cube_kept_place fails when the cube slides more than 5 mm in the drawer, just
above the expert's largest slide (4.0 mm). Per run and S1 variant this prints:

    task     the task without the slide limit: every other check passes and the arms hold still for the last 2 s
    slide    the cube's slide in mm over the task rollouts (median, 90th percentile, largest)
    @5/6/8   task and a slide under 5 mm (the stop rate: the reported success, also held still), 6 mm and 8 mm
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LIMITS = (5, 6, 8)  # mm


def rows(run: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in open(run / "episodes.jsonl"):
        r = json.loads(line)
        if r["variant"][0] == "S1":
            by[":".join(x for x in r["variant"] if x)].append(r)
    return by


def report(rs: list[dict]) -> dict:
    task = [all(v for k, v in r["checks"].items() if k != "cube_kept_place") and r.get("held_still", True) for r in rs]
    slide = np.array([r["cube_slide_mm"] for r, t in zip(rs, task) if t])
    out = {"n": len(rs), "task": sum(task)}
    out["slide"] = (np.median(slide), np.percentile(slide, 90), slide.max()) if len(slide) else None
    for mm in LIMITS:
        out[mm] = int((slide < mm).sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", type=Path, nargs="+", help="evaluation directories with episodes.jsonl")
    a = p.parse_args()
    print("| Run | Variant | Task | Slide mm (median, p90, max) | @5 mm | @6 mm | @8 mm |")
    print("|---|---|---|---|---|---|---|")
    for run in a.runs:
        for name, rs in rows(run).items():
            s = report(rs)
            slide = "—" if s["slide"] is None else ", ".join(f"{x:.1f}" for x in s["slide"])
            print(f"| {run} | {name} | {s['task']}/{s['n']} | {slide} | " + " | ".join(f"{s[mm]}/{s['n']}" for mm in LIMITS) + " |")


if __name__ == "__main__":
    main()

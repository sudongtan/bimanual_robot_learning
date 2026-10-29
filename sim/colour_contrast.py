"""Cube colour sweep on a recorded dataset (2data_collection.md, "Cube colours").

    python -m sim.colour_contrast data/a0               # report in data/a0/check/colours.txt
    python -m sim.colour_contrast data/a0 --stride 10   # every 10th frame (default 20)

For one successful episode per variant, re-render the overview from the recorded states with only the
cube's colour changed, for every candidate: the palettes in sim/randomize.py and the colours rejected
for them. Contrast = RGB distance (0-1 per channel) between the cube's mean rendered colour and each
surround (check_data.surround_geoms: the arms' light and dark parts, table, drawer, handle, cabinet),
median over frames. A colour's score is its weakest surround; check_data flags episodes below
MIN_CONTRAST. Also reports each eval colour's nominal RGB distance to the nearest training colour.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from sim import check_data as C
from sim import episode as E
from sim import randomize as RZ
from sim import record as R
from sim.scene import STATE_SPEC

# Measured and rejected (2data_collection.md, "Cube colours"); the old palette's colours among them.
REJECTED = {
    "orange": (0.95, 0.55, 0.10), "yellow": (0.95, 0.85, 0.15), "purple": (0.55, 0.30, 0.75),
    "pink": (0.95, 0.45, 0.70), "lime": (0.65, 0.90, 0.20), "black": (0.10, 0.10, 0.10),
    "white": (0.95, 0.95, 0.95), "grey": (0.50, 0.50, 0.50), "dark_green": (0.05, 0.40, 0.15),
    "sky": (0.45, 0.70, 0.95), "brown": (0.45, 0.25, 0.10), "indigo": (0.25, 0.20, 0.65),
    "mint": (0.55, 0.95, 0.75),
}
MIN_PX = 20  # cube or surround pixels needed to measure a frame


def sweep(root: Path, candidates: dict, stride: int) -> dict:
    """colour -> surround -> per-frame contrast, over one successful episode per variant."""
    log = [json.loads(line) for line in open(root / "sim" / "episodes.jsonl")]
    rand = C.randomization(json.loads((root / "sim" / "info.json").read_text()))
    seen, picks = set(), []
    for e in log:
        if e["success"] and tuple(e["variant"]) not in seen:
            seen.add(tuple(e["variant"]))
            picks.append(e)
    scores = defaultdict(lambda: defaultdict(list))
    for e in picks:
        sim, _ = E.new_sim(e["seed"], rand)
        model = sim.model
        data = mujoco.MjData(model)
        surrounds = C.surround_geoms(sim)
        cube = model.geom("cube").id
        rgb, seg = mujoco.Renderer(model, R.HEIGHT, R.WIDTH), mujoco.Renderer(model, R.HEIGHT, R.WIDTH)
        seg.enable_segmentation_rendering()
        with np.load(root / "sim" / e["states"]) as f:
            states = f["state"][::stride]
        for state in states:
            mujoco.mj_setState(model, data, state, STATE_SPEC)
            mujoco.mj_forward(model, data)
            seg.update_scene(data, camera="overview")
            ids = seg.render()
            geom = np.where(ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), ids[..., 0], -1)
            on_cube = geom == cube
            if on_cube.sum() < MIN_PX:
                continue
            masks = {k: np.isin(geom, g) for k, g in surrounds.items()}
            masks = {k: m for k, m in masks.items() if m.sum() >= MIN_PX}
            for name, colour in candidates.items():
                model.geom_rgba[cube, :3] = colour
                rgb.update_scene(data, camera="overview")
                img = rgb.render() / 255.0
                cube_mean = img[on_cube].mean(0)
                for k, m in masks.items():
                    scores[name][k].append(float(np.linalg.norm(cube_mean - img[m].mean(0))))
        rgb.close()
        seg.close()
        print(f"{':'.join(x for x in e['variant'] if x)} seed {e['seed']}: {len(states)} frames", flush=True)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--stride", type=int, default=20, help="frames between renders")
    a = p.parse_args()

    use = {**{c: "train" for c in RZ.TRAIN_COLOURS}, **{c: "eval" for c in RZ.EVAL_COLOURS}}
    candidates = {**RZ.TRAIN_COLOURS, **RZ.EVAL_COLOURS, **REJECTED}
    scores = sweep(a.root, candidates, a.stride)

    order = ["arm_light", "arm_dark", "table", "drawer", "handle", "cabinet"]
    rows = []
    for name in candidates:
        med = {k: float(np.median(scores[name][k])) for k in order if scores[name][k]}
        worst = min(med, key=med.get)
        rows.append((med[worst], name, worst, med))
    lines = [f"{'colour':11s} {'use':6s} {'worst':>6s} {'(surround)':11s} " + " ".join(f"{k:>9s}" for k in order)]
    for w, name, worst, med in sorted(rows, reverse=True):
        flag = "  < MIN_CONTRAST" if w < C.MIN_CONTRAST and name in use else ""
        lines.append(f"{name:11s} {use.get(name, '-'):6s} {w:6.2f} {worst:11s} "
                     + " ".join(f"{med.get(k, float('nan')):9.2f}" for k in order) + flag)
    lines.append("")
    lines.append("eval colour: nearest training colour (nominal RGB distance)")
    for e, c in RZ.EVAL_COLOURS.items():
        d = {t: float(np.linalg.norm(np.subtract(c, v))) for t, v in RZ.TRAIN_COLOURS.items()}
        t = min(d, key=d.get)
        lines.append(f"  {e}: {t} {d[t]:.2f}")

    out = a.root / "check"
    out.mkdir(exist_ok=True)
    (out / "colours.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out / 'colours.txt'}")


if __name__ == "__main__":
    main()

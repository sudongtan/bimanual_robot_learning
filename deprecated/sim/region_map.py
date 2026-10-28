"""Find a start region for the items (README 2.1 D).

    python -m sim.region_map ik        # where the fixed grasp exists at all, per tilt (minutes)
    python -m sim.region_map expert    # where the full Task 1 expert episode succeeds
    python -m sim.region_map validate  # random starts inside the chosen region

Maps cover the left arm's half only; the right arm is the left arm rotated 180°, and the
scene is symmetric under that rotation (plastic <-> metal side), so its map is the same
rotated. `validate` samples both arms.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

STEP = 0.02
XS = np.round(np.arange(-0.04, -0.2301, -STEP), 3)  # left half: middle band to table edge
YS = np.round(np.arange(-0.20, 0.2001, STEP), 3)
BASE_KEEP_OUT = 0.10
TABLE_HALF = (0.25, 0.23)
EDGE_MARGIN = 0.01  # item stays this far inside the table edge


# Chosen start region (README 2.1 D): four patches, |x| in X_RANGE on the picking arm's side and
# |y| in Y_RANGE in front or behind. Same patches for both items, so position doesn't reveal
# which item it is. From the expert map: every 2 cm grid cell inside passes for both items.
X_RANGE = (0.08, 0.14)
Y_RANGE = (0.16, 0.20)


def sample_start(rng: np.random.Generator, arm: str | None = None) -> tuple[float, float]:
    """Uniform start position in one of the patches (on `arm`'s side if given)."""
    sx = {"left": -1, "right": 1}[arm] if arm else rng.choice((-1, 1))
    sy = rng.choice((-1, 1))
    return float(sx * rng.uniform(*X_RANGE)), float(sy * rng.uniform(*Y_RANGE))


def spots(item: str | None = None):
    """Grid cells on the left half, outside the base keep-out and (per item) fully on the table."""
    from sim.check_requirements import ITEMS

    r = ITEMS[item]["width"] / 2 + EDGE_MARGIN if item else 0.0
    return [
        (x, y)
        for y in YS
        for x in XS
        if np.hypot(x + 0.2, y) >= BASE_KEEP_OUT and abs(x) <= TABLE_HALF[0] - r and abs(y) <= TABLE_HALF[1] - r
    ]


def _ik_cell(args):
    from sim.check_requirements import ITEMS
    from sim.expert import plan_grasp
    from sim.scene import TABLE_Z, Sim

    item, tilt, (x, y) = args
    sim = Sim(seed=None, noise=0)
    sim.clear_items()
    sim.place(item, (x, y, TABLE_Z + ITEMS[item]["half_h"] + 0.001))
    return plan_grasp(sim, "left", item, tilt=tilt) is not None


def _validate_cell(args):
    from sim.expert import task1_episode

    item, seed = args
    rng = np.random.default_rng(1000 + seed)
    return task1_episode(item, sample_start(rng), seed=seed)


def _expert_cell(args):
    from sim.expert import task1_episode

    item, xy, seed = args
    return task1_episode(item, xy, seed=seed)


def draw(ok: dict, title: str):
    print(f"\n{title}   (# = pass, . = fail, ' ' = base keep-out; left arm base at x=-0.20, y=0)")
    print("  y\\x " + "".join(f"{x * 100:4.0f}" for x in XS[::-1]))
    for y in YS[::-1]:
        row = "".join(f"{('#' if ok[(x, y)] else '.') if (x, y) in ok else ' ':>4}" for x in XS[::-1])
        print(f"{y * 100:+5.0f} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ik", "expert", "validate"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=60, help="random starts per item for validate")
    args = ap.parse_args()
    cells = spots()
    with ProcessPoolExecutor(args.workers) as pool:
        if args.stage == "validate":
            for item in ("block", "can"):
                res = list(pool.map(_validate_cell, [(item, k) for k in range(args.n)], chunksize=2))
                fails = [r for r in res if not r["success"]]
                arms = {a: sum(1 for r in res if r["picker"] == a) for a in ("left", "right")}
                print(f"{item}: {len(res) - len(fails)}/{len(res)} succeed (picker left {arms['left']}, right {arms['right']})")
                for r in fails:
                    print("   fail", {k: r.get(k) for k in ("x", "y", "picker", "result", "held", "in_drawer", "arm_contacts")})
                print(f"   max joint jump {max(r.get('max_jump', 0) for r in res):.2f} rad, "
                      f"max push during approach {max(r.get('disturbed_cm', 0) for r in res):.1f} cm")
        if args.stage == "expert":
            for item in ("block", "can"):
                cells = spots(item)
                res = list(pool.map(_expert_cell, [(item, c, 0) for c in cells], chunksize=2))
                ok = dict(zip(cells, (r["success"] for r in res)))
                draw(ok, f"{item}: Task 1 expert episode succeeds ({sum(ok.values())}/{len(ok)})")
                reasons = {}
                for r in res:
                    if not r["success"]:
                        k = r["result"] if r["result"] != "ok" else (
                            "dropped" if not r["held"] else "missed drawer" if not r["in_drawer"]
                            else "arm contact" if r["arm_contacts"] else "drawer not open")
                        reasons[k] = reasons.get(k, 0) + 1
                print("failures:", reasons)
                pushed = [r for r in res if r.get("disturbed_cm", 0) > 1.0]
                print("item pushed > 1 cm during approach:", [(r["x"], r["y"], r["disturbed_cm"]) for r in pushed])
                jumps = [r for r in res if r.get("max_jump", 0) > 0.3]
                print("IK branch jumps > 0.3 rad:", [(r["x"], r["y"], r["max_jump"]) for r in jumps])
        if args.stage == "ik":
            for item, tilts in (("block", (0, 10, 20, 30)), ("can", (10, 20, 30))):
                for tilt in tilts:
                    res = list(pool.map(_ik_cell, [(item, tilt, c) for c in cells], chunksize=4))
                    ok = dict(zip(cells, res))
                    draw(ok, f"{item}, tilt {tilt}°: grasp + pre-grasp reachable, open gripper clear ({sum(res)}/{len(res)})")


if __name__ == "__main__":
    main()

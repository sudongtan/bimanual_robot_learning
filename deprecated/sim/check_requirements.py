"""Check the sorting-table scene against README section 2.1 (requirements A–E).

    uv run python -m sim.check_requirements            # full run, a few minutes
    uv run python -m sim.check_requirements --quick    # fewer seeds and positions
    uv run python -m sim.check_requirements --images out/   # also save contact sheets

Motions are scripted (straight lines solved with IK), so a failure means either the
scene or the script; each check reports enough to tell which. The C and A2 checks
start from a grasp pose set directly (no approach motion), so they measure whether an
item can be held and carried, not whether this script can approach it cleanly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sim.scene import (
    ARMS,
    DOWN,
    DRAWER_PULL,
    GRIPPER_CLOSED,
    TABLE_Z,
    Sim,
    grasp_target,
    load_model,
    Arm,
    solve_ik,
)

Z = np.array([0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])

# Pass thresholds (proposed in the README discussion; adjust to taste).
MIN_DRAWER_OPEN = 0.10  # m: leaves ~8.5 cm of interior past the table edge (can is 6.8 cm)
HANDLE_GRIP_DEPTH = 0.01  # m: bar sits this far inside the jaws; 2 cm limits the pull to ~10.5 cm
MIDDLE_BAND = 0.04  # m: items stay |x| >= this (README 2.1 D)
BASE_KEEP_OUT = 0.10  # m: items stay this far from either arm base (spawn overlap otherwise)

ITEMS = {
    "block": dict(width=0.03, half_h=0.025, drawer="plastic", open=0.8, carry_z=0.43),
    "can": dict(width=0.04, half_h=0.03, drawer="metal", open=1.0, carry_z=0.44),
}
DRAWER_SIGN = {"plastic": 1, "metal": -1}


# --- helpers ---------------------------------------------------------------


def radial(sim: Sim, arm: str, p) -> np.ndarray:
    """Horizontal unit vector from the arm's pan axis toward p."""
    r = np.asarray(p, float) - sim.data.body(f"{sim.arms[arm].name}/shoulder").xpos
    r[2] = 0
    return r / np.linalg.norm(r)


def grasp_candidates(item: str, rad: np.ndarray):
    """(label, fingertip height above table, pointing, jaw_perp), most preferred first."""
    face = np.array([0.0, 1, 0]) if abs(rad[0]) > abs(rad[1]) else X  # jaws close across block faces
    tilts = range(0, 61, 10)
    for t in tilts:  # top grasp: point down, tilted t° outward
        yield f"top, {t}° tilt", 2 * ITEMS[item]["half_h"] - 0.015, np.cos(np.deg2rad(t)) * DOWN + np.sin(np.deg2rad(t)) * rad, face


def set_grasp_pose(sim: Sim, arm: str, item: str) -> str | None:
    """Put `arm` around `item` (where it currently stands) with the gripper open.

    Returns a description of the grasp, or None if no pose is reachable without the
    open gripper touching the item or the table.
    """
    a, cfg = sim.arms[arm], ITEMS[item]
    centre = sim.body_pos(item)
    rad = radial(sim, arm, centre)
    saved = sim.data.qpos.copy()
    scratch = mujoco.MjData(sim.model)
    for label, h, pt, jaw in grasp_candidates(item, rad):
        c = np.array([centre[0], centre[1], TABLE_Z + h])
        r1 = solve_ik(sim.model, a, c, point=pt, jaw_perp=jaw, restarts=8)
        if not r1.ok(point_tol=np.deg2rad(8)):
            continue
        scratch.qpos[:] = saved
        scratch.qpos[a.qadr] = r1.q
        mujoco.mj_kinematics(sim.model, scratch)
        g = grasp_target(c, scratch.site_xmat[a.site].reshape(3, 3)[:, 2], cfg["width"])
        r2 = solve_ik(sim.model, a, g, point=pt, jaw_perp=jaw, q0=r1.q)
        if not r2.ok(point_tol=np.deg2rad(8)):
            continue
        sim.data.qpos[:] = saved
        sim.data.qpos[a.qadr] = r2.q
        sim.data.ctrl[a.act] = r2.q
        sim.data.qpos[a.grip_qadr] = sim.data.ctrl[a.grip_act] = cfg["open"]
        mujoco.mj_forward(sim.model, sim.data)
        if sim.touching(arm, item) or sim.touching(arm, "table") or sim.arms_touching():
            continue
        return label
    sim.data.qpos[:] = saved
    mujoco.mj_forward(sim.model, sim.data)
    return None


def inside(sim: Sim, site: str, p, margin: float = 0.01) -> bool:
    c, R, hs = sim.data.site(site).xpos, sim.data.site(site).xmat.reshape(3, 3), sim.model.site(site).size
    return bool(np.all(np.abs(R.T @ (np.asarray(p) - c)) <= hs + margin))


def visible_pixels(sim: Sim, renderer: mujoco.Renderer, camera: str, body: str) -> int:
    """Pixels of `body` in a segmentation render from `camera` (224 x 224, as for training)."""
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera=camera)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    bid = sim.model.body(body).id
    geoms = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    return int(np.isin(seg[..., 0], geoms)[seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM].sum())


class Recorder:
    """Collects frames every `every` steps for a contact sheet."""

    def __init__(self, sim: Sim, enabled: bool, lookat=(0, 0, 0.35), azimuth=90.0, every=50):
        self.frames, self.n, self.every = [], 0, every
        self.renderer = mujoco.Renderer(sim.model, 300, 400) if enabled else None
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = lookat, 0.95, azimuth, -30

    def __call__(self, sim: Sim):
        self.n += 1
        if self.renderer is not None and self.n % self.every == 0:
            self.renderer.update_scene(sim.data, camera=self.cam)
            self.frames.append(self.renderer.render())

    def save(self, path: Path | None, cols: int = 4, rows: int = 2):
        if self.renderer is None or path is None or not self.frames:
            return
        import imageio.v3 as iio

        idx = np.linspace(0, len(self.frames) - 1, cols * rows).astype(int)
        grid = [np.concatenate([self.frames[i] for i in idx[r * cols : (r + 1) * cols]], 1) for r in range(rows)]
        iio.imwrite(path, np.concatenate(grid, 0))


# --- A1: arms don't interfere at rest ----------------------------------------


def check_home(seeds: int) -> dict:
    sim = Sim(seed=0)
    worst, touching = 1.0, 0
    for seed in range(seeds):
        sim.reset(seed)
        worst = min(worst, sim.arm_clearance(distmax=0.5))
        touching += sim.arms_touching()
    tips = np.linalg.norm(sim.tip("left") - sim.tip("right"))
    return dict(seeds=seeds, min_clearance_cm=round(worst * 100, 1), seeds_touching=touching, tip_gap_cm=round(tips * 100, 1))


# --- B: both arms open both drawers --------------------------------------------


def drawer_via(arm: str, drawer: str) -> np.ndarray:
    """Waypoint on the opener's own side, toward its drawer. Going straight from home in joint
    space sweeps through the other arm, which leans toward the opposite drawer at home."""
    return np.array([0.12 if arm == "right" else -0.12, DRAWER_SIGN[drawer] * 0.10, 0.40])


def open_drawer(sim: Sim, arm: str, drawer: str, rec=None) -> dict:
    """Grasp the handle, pull the full travel, release and back off. Returns measurements."""
    sg = DRAWER_SIGN[drawer]
    h = np.array([0, sg * 0.285, 0.395])  # handle bar centre (README 2.2)
    rad = radial(sim, arm, h)
    err = []
    sim.grip(arm, 0.7, 0.3, rec)
    with sim.noisy(arm, rec):
        sim.move_to(arm, drawer_via(arm, drawer), 1.0, on_step=rec)
        err.append(sim.move_to(arm, h - 0.07 * rad + 0.02 * Z, 1.2, point=rad, jaw_perp=X, on_step=rec).pos_err)
    err.append(sim.move_line(arm, h + HANDLE_GRIP_DEPTH * rad, 0.8, point=rad, jaw_perp=X, on_step=rec))
    sim.grip(arm, GRIPPER_CLOSED, 0.5, rec)
    gripped = bool(sim.touching(arm, f"{drawer}_drawer"))
    # Aim 1 cm short of full travel: with the bar 1 cm in the jaws the arm reaches ~11.5 cm.
    pull_err = sim.move_line(arm, sim.tip(arm) + [0, sg * (DRAWER_PULL - 0.01), 0], 2.0, jaw_perp=X, on_step=rec)
    opened = sim.drawer_opening(drawer)
    sim.grip(arm, 0.9, 0.4, rec)
    # Clear the fixed jaw off the bar before backing out, or it drags the drawer shut.
    sim.move_line(arm, sim.tip(arm) - 0.025 * sim.jaw_axis(arm), 0.4, on_step=rec)
    sim.move_line(arm, sim.tip(arm) - 0.08 * sim.pointing(arm), 0.8, on_step=rec)
    sim.move_line(arm, sim.tip(arm) + 0.05 * Z, 0.4, on_step=rec)
    return dict(gripped=gripped, approach_ik_err=max(err), pull_ik_err=pull_err, opened=opened)


def drawer_trial(arm: str, drawer: str, seed: int, image: Path | None = None) -> dict:
    sim = Sim(seed=seed)
    sim.clear_items()
    rec = Recorder(sim, image is not None, lookat=(0, DRAWER_SIGN[drawer] * 0.1, 0.35), azimuth=0 if arm == "right" else 180)
    m = open_drawer(sim, arm, drawer, rec)
    sim.step(500, rec)
    rec.save(image)
    return dict(
        arm=arm,
        drawer=drawer,
        seed=seed,
        gripped=m["gripped"],
        approach_ik_err_cm=round(100 * m["approach_ik_err"], 1),
        pull_ik_err_cm=round(100 * m["pull_ik_err"], 1),
        open_while_held_cm=round(100 * m["opened"], 1),
        open_after_release_cm=round(100 * sim.drawer_opening(drawer), 1),
        arm_contact_steps=sim.arm_contacts,
        min_open_ok=min(m["opened"], sim.drawer_opening(drawer)) >= MIN_DRAWER_OPEN,
    )


def pull_limit() -> list[dict]:
    """Kinematic: how far each arm can follow the handle, versus how deep the bar sits in the jaws."""
    model = load_model()
    sim = Sim(seed=0, noise=0)
    rows = []
    for arm in ARMS:
        a = Arm.from_model(model, arm)
        for drawer, sg in DRAWER_SIGN.items():
            h = np.array([0, sg * 0.285, 0.395])
            rad = radial(sim, arm, h)
            row = dict(arm=arm, drawer=drawer)
            for depth in (0.0, 0.01, 0.02):
                best = 0.0
                for seed in range(15):
                    g = h + depth * rad
                    q0 = np.random.default_rng(seed).uniform(a.lo, a.hi)
                    s = solve_ik(model, a, g, point=rad, jaw_perp=X, q0=q0)
                    if not s.ok():
                        continue
                    q, reach = s.q, 0.0
                    for k in range(1, 31):
                        t = solve_ik(model, a, g + [0, sg * 0.005 * k, 0], jaw_perp=X, q0=q)
                        if not t.ok(pos_tol=0.003):
                            break
                        q, reach = t.q, 0.005 * k
                    best = max(best, reach)
                row[f"depth_{round(depth * 100)}cm"] = round(best * 100, 1)
            rows.append(row)
    return rows


# --- C: pick up and carry without dropping --------------------------------------


def item_positions(arm: str, quick: bool) -> list[tuple[float, float]]:
    """Spawn spots on the arm's own half that respect the middle band and base keep-out."""
    sx = -1 if arm == "left" else 1
    xs = (0.06, 0.12) if quick else (0.05, 0.09, 0.13)
    ys = (-0.12, 0.0, 0.12) if quick else (-0.16, -0.08, 0.0, 0.08, 0.16)
    out = []
    for x in xs:
        for y in ys:
            p = (sx * x, y)
            if min(np.hypot(p[0] - bx, p[1]) for bx in (-0.2, 0.2)) >= BASE_KEEP_OUT and abs(p[0]) >= MIDDLE_BAND:
                out.append(p)
    return out


def carry_trial(arm: str, item: str, xy, seed: int = 0, renderer=None, image: Path | None = None) -> dict:
    sim = Sim(seed=seed)
    sim.clear_items()
    cfg = ITEMS[item]
    drawer = cfg["drawer"]
    sg = DRAWER_SIGN[drawer]
    sim.data.joint(f"{drawer}_drawer").qpos[0] = MIN_DRAWER_OPEN
    sim.place(item, (xy[0], xy[1], TABLE_Z + cfg["half_h"] + 0.001))
    sim.step(50)
    row = dict(arm=arm, item=item, x=xy[0], y=xy[1])
    if sim.touching("left", item) or sim.touching("right", item):
        return row | dict(result="spawn overlaps an arm")
    grasp = set_grasp_pose(sim, arm, item)
    if grasp is None:
        return row | dict(result="no grasp pose")
    rec = Recorder(sim, image is not None, azimuth=90 - sg * 90 + 30)
    m = carry_to_drawer(sim, arm, item, rec)
    rec.save(image)
    vis = {}
    if renderer is not None and m["in_drawer"]:
        vis = {f"px_{cam.split('/')[0]}": visible_pixels(sim, renderer, cam, item) for cam in ("overview", "left/wrist_cam", "right/wrist_cam")}
    return row | dict(
        result="ok",
        grasp=grasp,
        held=m["held"],
        slip_cm=round(100 * m["slip"], 1),
        ik_err_cm=round(100 * m["ik_err"], 1),
        in_drawer=m["in_drawer"],
        arm_contact_steps=sim.arm_contacts,
        **vis,
    )


def carry_to_drawer(sim: Sim, arm: str, item: str, rec=None) -> dict:
    """From a grasp pose: close, lift, carry above the item's drawer, release, back off."""
    cfg = ITEMS[item]
    drawer = cfg["drawer"]
    sg = DRAWER_SIGN[drawer]
    sim.step(100, rec)
    sim.grip(arm, GRIPPER_CLOSED, 0.6, rec)
    offset = sim.body_pos(item) - sim.tip(arm)
    slip = [0.0]

    def watch(s: Sim):
        slip[0] = max(slip[0], float(np.linalg.norm(s.body_pos(item) - s.tip(arm) - offset)))
        if rec is not None:
            rec(s)

    err = sim.move_line(arm, sim.tip(arm) + 0.12 * Z, 0.8, on_step=watch)
    err = max(err, sim.move_line(arm, np.array([0, sg * 0.27, cfg["carry_z"]]), 1.2, on_step=watch))
    sim.step(250, watch)
    held = bool(sim.touching(arm, item)) and sim.body_pos(item)[2] > TABLE_Z + 0.05
    sim.grip(arm, cfg["open"], 0.5, rec)
    sim.move_line(arm, sim.tip(arm) + [0, -sg * 0.08, 0.04], 0.6, on_step=rec)
    sim.step(750, rec)
    return dict(held=held, slip=slip[0], ik_err=err, in_drawer=inside(sim, f"{drawer}_interior", sim.body_pos(item)))


# --- A1 (both arms at once): Task 1 end to end -------------------------------------------


def task1_trial(item: str, xy, seed: int = 0, image: Path | None = None) -> dict:
    """The closer arm carries `item` to its drawer while the other arm opens that drawer first."""
    picker = "left" if xy[0] < 0 else "right"
    opener = "right" if picker == "left" else "left"
    drawer = ITEMS[item]["drawer"]
    sim = Sim(seed=seed)
    sim.clear_items()
    sim.place(item, (xy[0], xy[1], TABLE_Z + ITEMS[item]["half_h"] + 0.001))
    sim.step(50)
    row = dict(item=item, x=xy[0], y=xy[1], picker=picker, opener=opener, drawer=drawer)
    rec = Recorder(sim, image is not None, azimuth=90 - DRAWER_SIGN[drawer] * 90 + 30)
    o = open_drawer(sim, opener, drawer, rec)
    # The opener parks over the drop zone after releasing; send it home before the picker arrives.
    sim.move_line(opener, sim.tip(opener) + 0.04 * Z, 0.4, on_step=rec)
    sim.move_to(opener, drawer_via(opener, drawer), 1.0, on_step=rec)
    sim.move_joints({opener: sim.home}, 1.5, on_step=rec)
    contacts_opening = sim.arm_contacts
    if set_grasp_pose(sim, picker, item) is None:
        return row | dict(result="no grasp pose")
    c = carry_to_drawer(sim, picker, item, rec)
    rec.save(image, rows=3)
    return row | dict(
        result="ok",
        opened_cm=round(100 * sim.drawer_opening(drawer), 1),
        held=c["held"],
        in_drawer=c["in_drawer"],
        arm_contact_steps_opening=contacts_opening,
        arm_contact_steps_carrying=sim.arm_contacts - contacts_opening,
        min_open_ok=o["opened"] >= MIN_DRAWER_OPEN,
    )


# --- A2: handover ----------------------------------------------------------------


def handover_trial(giver: str, seed: int, meet=(0.0, 0.08, 0.42), image: Path | None = None) -> dict:
    # On the centre line (y = 0) the arms are too close to fold around each other: the taker
    # would need joints at their limits. 8 cm off it works in both directions.
    taker = "right" if giver == "left" else "left"
    sx = -1 if giver == "left" else 1
    sim = Sim(seed=seed)
    sim.clear_items()
    sim.place("block", (sx * 0.09, -sx * 0.08, TABLE_Z + 0.026))  # same spot in each giver's own frame
    sim.step(50)
    row = dict(giver=giver, seed=seed)
    if set_grasp_pose(sim, giver, "block") is None:
        return row | dict(result="giver has no grasp pose")
    rec = Recorder(sim, image is not None, lookat=(0, 0, 0.4), azimuth=90, every=40)
    sim.grip(giver, GRIPPER_CLOSED, 0.6, rec)
    sim.move_line(giver, sim.tip(giver) + 0.08 * Z, 0.6, on_step=rec)
    err = sim.move_line(giver, np.asarray(meet), 1.2, on_step=rec)
    sim.step(200, rec)
    if not sim.touching(giver, "block"):
        rec.save(image)
        return row | dict(result="giver dropped the block on the way")
    # Taker: side grasp of the block's lower half, jaws closing across its faces.
    Rb = sim.data.body("block").xmat.reshape(3, 3)
    target = sim.body_pos("block") - 0.012 * Rb[:, 2]
    rad = radial(sim, taker, target)
    back = 0.05  # pre-grasp distance behind the grasp, along the pointing direction
    pt = None
    for t in range(0, 91, 15):
        cand = np.cos(np.deg2rad(t)) * rad + np.sin(np.deg2rad(t)) * DOWN
        q1 = solve_ik(sim.model, sim.arms[taker], target, point=cand, jaw_perp=Rb[:, 2], restarts=8)
        pre = solve_ik(sim.model, sim.arms[taker], target - back * cand, point=cand, jaw_perp=Rb[:, 2], q0=q1.q, restarts=8)
        if q1.ok(point_tol=np.deg2rad(8)) and pre.ok(point_tol=np.deg2rad(8)):
            pt, tilt = cand, t
            break
    if pt is None:
        rec.save(image)
        return row | dict(result="taker has no grasp pose at the meeting point")
    contacts_before = sim.arm_contacts
    sim.grip(taker, 0.8, 0.3, rec)
    err = max(err, sim.move_to(taker, target - back * pt, 1.6, point=pt, jaw_perp=Rb[:, 2], on_step=rec).pos_err)
    g = grasp_target(target, sim.jaw_axis(taker), ITEMS["block"]["width"])
    err = max(err, sim.move_line(taker, g - back * pt, 0.3, point=pt, jaw_perp=Rb[:, 2], on_step=rec))
    err = max(err, sim.move_line(taker, g, 0.6, point=pt, jaw_perp=Rb[:, 2], on_step=rec))
    sim.grip(taker, GRIPPER_CLOSED, 0.6, rec)
    both = bool(sim.touching(giver, "block")) and bool(sim.touching(taker, "block"))
    sim.grip(giver, 0.8, 0.4, rec)
    sim.move_line(giver, sim.tip(giver) + 0.06 * Z + [sx * 0.06, 0, 0], 0.8, on_step=rec)
    sim.move_line(taker, sim.tip(taker) + 0.03 * Z - [sx * 0.05, 0, 0], 0.8, on_step=rec)
    sim.step(500, rec)
    rec.save(image, rows=3)
    taker_holds = bool(sim.touching(taker, "block")) and not sim.touching(giver, "block") and sim.body_pos("block")[2] > TABLE_Z + 0.05
    return row | dict(
        result="ok",
        taker_grasp=f"side, {tilt}° down",
        both_holding=both,
        taker_holds_after=taker_holds,
        ik_err_cm=round(100 * err, 1),
        arm_contact_steps_transfer=sim.arm_contacts - contacts_before,
        arm_contact_steps_total=sim.arm_contacts,
    )


# --- D: reach over the tabletop -----------------------------------------------------


def reach_map(step: float = 0.04) -> dict:
    """Block top-grasp reach (≤ 60° tilt) over the tabletop, per arm."""
    model = load_model()
    arms = {a: Arm.from_model(model, a) for a in ARMS}
    xs = np.arange(-0.22, 0.2201, step)
    ys = np.arange(-0.20, 0.2001, step)
    grid = {a: np.zeros((len(ys), len(xs)), bool) for a in ARMS}
    for arm, a in arms.items():
        base = np.array([-0.2 if arm == "left" else 0.2, 0, 0])
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                p = np.array([x, y, TABLE_Z + 0.035])
                r = p - base
                r[2] = 0
                if np.linalg.norm(r) < 0.03:
                    continue
                r /= np.linalg.norm(r)
                for t in range(0, 61, 15):
                    pt = np.cos(np.deg2rad(t)) * DOWN + np.sin(np.deg2rad(t)) * r
                    if solve_ik(model, a, p, point=pt, restarts=6).ok(point_tol=np.deg2rad(5)):
                        grid[arm][i, j] = True
                        break
    L, R = grid["left"], grid["right"]
    n = L.size
    return dict(
        cells=n,
        both_pct=round(100 * (L & R).sum() / n),
        only_left_pct=round(100 * (L & ~R).sum() / n),
        only_right_pct=round(100 * (~L & R).sum() / n),
        neither_pct=round(100 * (~L & ~R).sum() / n),
        xs=xs,
        ys=ys,
        grid=grid,
    )


# --- report -----------------------------------------------------------------------------


def table(rows: list[dict], keys: list[str] | None = None) -> str:
    if not rows:
        return "(none)\n"
    keys = keys or list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer seeds and item positions")
    ap.add_argument("--images", type=Path, help="directory for contact-sheet PNGs")
    args = ap.parse_args()
    img = args.images
    if img:
        img.mkdir(parents=True, exist_ok=True)
    seeds = 2 if args.quick else 3
    out = []
    say = lambda s="": (print(s, flush=True), out.append(s))

    say("# Scene requirement check\n")
    home = check_home(10 if args.quick else 50)
    say("## A1. Arms don't interfere: home pose with start noise\n")
    say(table([home]))

    drawers = [
        drawer_trial(arm, drawer, seed, (img / f"B_{arm}_{drawer}.png") if img and seed == 0 else None)
        for seed in range(seeds)
        for arm in ARMS
        for drawer in DRAWER_SIGN
    ]
    say(f"## B. Both arms open both drawers (pass: ≥ {MIN_DRAWER_OPEN * 100:.0f} cm open and still open after release, no arm contact)\n")
    say(table(drawers))
    say("### B. Kinematic pull limit (cm of travel the fingertip can follow) vs. how deep the bar sits in the jaws\n")
    say(table(pull_limit()))

    renderer = mujoco.Renderer(load_model(), 224, 224)
    carries = []
    for arm in ARMS:
        for item in ITEMS:
            for k, xy in enumerate(item_positions(arm, args.quick)):
                path = (img / f"C_{arm}_{item}.png") if img and k == 1 else None
                carries.append(carry_trial(arm, item, xy, renderer=renderer, image=path))
    say("## C. Pick up, carry above the drawer, drop in (pass: still held above the drawer)\n")
    say(f"Spots on each arm's own half, |x| ≥ {MIDDLE_BAND * 100:.0f} cm, ≥ {BASE_KEEP_OUT * 100:.0f} cm from both bases. px_* = item pixels at 224×224 once in the drawer (E).\n")
    say(table(carries))

    task1 = [
        task1_trial(item, xy, image=(img / f"A1_task1_{arm}_{item}.png") if img and k == 0 else None)
        for item in ITEMS
        for arm in ARMS
        for k, xy in enumerate(item_positions(arm, quick=True))
    ]
    say("## A1. Both arms in one episode: Task 1 (other arm opens the drawer and returns home, closer arm drops the item in)\n")
    say(table(task1))

    handovers = [
        handover_trial(giver, seed, image=(img / f"A2_{giver}.png") if img and seed == 0 else None)
        for giver in ARMS
        for seed in range(seeds)
    ]
    say("## A2. Handover (block, meeting point (0, 0.08, 0.42))\n")
    say(table(handovers))

    reach = reach_map(0.08 if args.quick else 0.04)
    say("## D. Block top-grasp reach over the tabletop (≤ 60° tilt)\n")
    say(table([{k: v for k, v in reach.items() if k not in ("xs", "ys", "grid")}]))
    say("```")
    say("y\\x    " + "".join(f"{x:+6.2f}" for x in reach["xs"]))
    for i, y in reversed(list(enumerate(reach["ys"]))):
        cells = "".join(
            f"{'B' if reach['grid']['left'][i, j] and reach['grid']['right'][i, j] else 'L' if reach['grid']['left'][i, j] else 'R' if reach['grid']['right'][i, j] else '.':>6}"
            for j in range(len(reach["xs"]))
        )
        say(f"{y:+.2f} {cells}")
    say("B = both arms, L/R = one arm, . = neither (arm bases at x = ±0.20, y = 0)")
    say("```\n")

    # --- summary
    ok_drawer = {(r["arm"], r["drawer"]) for r in drawers if r["min_open_ok"] and r["arm_contact_steps"] == 0}
    b_runs = sum(1 for r in drawers if r["min_open_ok"] and r["arm_contact_steps"] == 0)
    c_ok = [r for r in carries if r.get("result") == "ok" and r["held"]]
    c_runs = [r for r in carries if r.get("result") == "ok"]
    h_ok = [r for r in handovers if r.get("result") == "ok" and r["taker_holds_after"]]
    t1_runs = [r for r in task1 if r.get("result") == "ok"]
    t1_ok = [r for r in t1_runs if r["in_drawer"] and r["arm_contact_steps_opening"] + r["arm_contact_steps_carrying"] == 0]
    all_trials = drawers + c_runs + [r for r in handovers if r.get("result") == "ok"]
    contact_free = sum(1 for r in all_trials if r.get("arm_contact_steps", r.get("arm_contact_steps_total", 0)) == 0)
    contact_free += sum(1 for r in t1_runs if r["arm_contact_steps_opening"] + r["arm_contact_steps_carrying"] == 0)
    all_trials += t1_runs
    say("## Summary\n")
    say(table([
        dict(req="A1 arms don't interfere", result=f"home: {home['seeds_touching']}/{home['seeds']} seeds touching, min gap {home['min_clearance_cm']} cm; trials without arm contact: {contact_free}/{len(all_trials)}"),
        dict(req="Task 1 end to end", result=f"{len(t1_ok)}/{len(task1)} episodes put the item in the drawer with no arm contact"),
        dict(req="A2 handover", result=f"{len(h_ok)}/{len(handovers)} handovers end with the taker holding the block"),
        dict(req="B both arms open both drawers", result=f"{len(ok_drawer)}/4 arm-drawer pairs pass; {b_runs}/{len(drawers)} runs"),
        dict(req="C pick up and carry", result=" · ".join(
            f"{item}: {sum(1 for r in c_ok if r['item'] == item)}/{sum(1 for r in c_runs if r['item'] == item)} held "
            f"({sum(1 for r in carries if r['item'] == item and r.get('result') != 'ok')} spots without a grasp pose)"
            for item in ITEMS)),
        dict(req="D reachable by at least one arm", result=f"{100 - reach['neither_pct']}% of the tabletop grid (both arms: {reach['both_pct']}%)"),
    ]))
    if img:
        (img / "report.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()

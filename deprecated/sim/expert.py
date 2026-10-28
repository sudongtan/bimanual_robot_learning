"""Privileged scripted expert for Task 1 (ACT / ManiSkill / RoboTwin style).

Reads item poses from the simulator. One fixed grasp per item, defined relative to the
picking arm's reach direction (so both arms use the same grasp):
    block: from above, tilted BLOCK_TILT outward from vertical, jaws across two faces,
           3.5 cm above the table (straight down is out of reach at the pre-grasp height)
    can:   from above, tilted CAN_TOP_TILT outward, fingertips 2 cm below the top, jaws
           across the diameter. A side grasp (CAN_STYLE = "side") pushed or dropped the can
           at every spot tried: coming in steeply, the gripper body sweeps over the rim.
Every motion is continuous from home (no teleporting): the opener opens the drawer and
returns home, then the picker goes pre-grasp -> straight in -> close -> lift -> carry ->
release -> home.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.check_requirements import DRAWER_SIGN, ITEMS, X, Z, drawer_via, inside, open_drawer, radial
from sim.scene import DOWN, GRIPPER_CLOSED, TABLE_Z, Sim, grasp_target, solve_ik

BLOCK_TILT = 10  # degrees outward from vertical; set from the grasp-exists map (sim/region_map.py)
CAN_STYLE = "top"  # "top" or "side"
CAN_TILT = 60  # side grasp: degrees below horizontal; set from the grasp-exists map
CAN_TOP_TILT = 20  # top grasp: degrees outward from vertical
PRE_GRASP = 0.06  # m back from the grasp along the pointing direction
HOVER = 0.06  # m above the pre-grasp, reached in joint space before descending
DEMO_NOISE = 0.008  # rad, σ of smooth demonstration noise on free-space moves (fingertip ≈ 3 mm RMS, < 1 cm max)
CARRY_TILT = 30  # degrees outward from vertical while carrying and dropping
POINT_TOL = np.deg2rad(8)


@dataclass
class GraspPlan:
    point: np.ndarray  # gripper pointing direction
    jaw_perp: np.ndarray  # jaws close across this direction's normal plane
    grasp: np.ndarray  # fingertip target at the grasp
    pre: np.ndarray  # fingertip target before going straight in
    q_grasp: np.ndarray


def grasp_style(item: str, rad: np.ndarray, tilt: float | None = None):
    """Fixed (fingertip height, pointing, jaw_perp) for an item, given the arm's reach direction."""
    if item == "block":
        face = np.array([0.0, 1, 0]) if abs(rad[0]) > abs(rad[1]) else X
        t = np.deg2rad(BLOCK_TILT if tilt is None else tilt)
        return 0.035, np.cos(t) * DOWN + np.sin(t) * rad, face
    if CAN_STYLE == "top":  # from above, fingertips 2 cm below the top, jaws across the diameter
        t = np.deg2rad(CAN_TOP_TILT if tilt is None else tilt)
        return 2 * ITEMS["can"]["half_h"] - 0.02, np.cos(t) * DOWN + np.sin(t) * rad, np.cross(Z, rad)
    t = np.deg2rad(CAN_TILT if tilt is None else tilt)
    return 0.055, np.cos(t) * rad + np.sin(t) * DOWN, Z


def _hits(model, data, arm_geoms: set[int], bodies: set[int], geoms: set[int]) -> bool:
    """Any contact between `arm_geoms` and a geom of `bodies` or in `geoms`."""
    for c in data.contact[: data.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a in arm_geoms and (model.geom_bodyid[b] in bodies or b in geoms):
                return True
    return False


def plan_grasp(sim: Sim, arm: str, item: str, tilt: float | None = None) -> GraspPlan | None:
    """Grasp for `item` where it stands now, or None if the fixed style can't reach it
    (grasp or pre-grasp out of reach, or the open gripper would touch the item, table or other arm)."""
    a, cfg = sim.arms[arm], ITEMS[item]
    centre = sim.body_pos(item)
    rad = radial(sim, arm, centre)
    h, pt, jaw = grasp_style(item, rad, tilt)
    c = np.array([centre[0], centre[1], TABLE_Z + h])
    r1 = solve_ik(sim.model, a, c, point=pt, jaw_perp=jaw, restarts=8)
    if not r1.ok(point_tol=POINT_TOL):
        return None
    scratch = mujoco.MjData(sim.model)
    scratch.qpos[:] = sim.data.qpos
    scratch.qpos[a.qadr] = r1.q
    mujoco.mj_kinematics(sim.model, scratch)
    g = grasp_target(c, scratch.site_xmat[a.site].reshape(3, 3)[:, 2], cfg["width"])
    r2 = solve_ik(sim.model, a, g, point=pt, jaw_perp=jaw, q0=r1.q)
    pre = g - PRE_GRASP * pt
    r3 = solve_ik(sim.model, a, pre, point=pt, jaw_perp=jaw, q0=r2.q, restarts=4)
    if not (r2.ok(point_tol=POINT_TOL) and r3.ok(point_tol=POINT_TOL)):
        return None
    other = sim.arms["right" if arm == "left" else "left"].geoms
    bodies = {sim.model.body(item).id, sim.model.body("table").id}
    for q in (r2.q, r3.q):
        scratch.qpos[:] = sim.data.qpos
        scratch.qpos[a.qadr] = q
        scratch.qpos[a.grip_qadr] = cfg["open"]
        mujoco.mj_forward(sim.model, scratch)
        if _hits(sim.model, scratch, a.geoms, bodies, other):
            return None
    return GraspPlan(point=pt, jaw_perp=jaw, grasp=g, pre=pre, q_grasp=r2.q)


def go_home(sim: Sim, arm: str, rec=None, via=None, duration: float = 1.5):
    """Lift clear, optionally pass a waypoint, then a joint-space move home (smooth; no IK on the way)."""
    sim.move_line(arm, sim.tip(arm) + 0.04 * Z, 0.4, on_step=rec)
    with sim.noisy(arm, rec):
        if via is not None:
            sim.move_to(arm, via, 1.0, on_step=rec)
        sim.move_joints({arm: sim.home}, duration, on_step=rec)


def pick_and_place(sim: Sim, arm: str, item: str, plan: GraspPlan, rec=None) -> dict:
    """Continuous pick from the current pose, carry above the item's drawer, drop, go home."""
    cfg = ITEMS[item]
    sg = DRAWER_SIGN[cfg["drawer"]]
    start = sim.body_pos(item)
    sim.data.ctrl[sim.arms[arm].grip_act] = cfg["open"]
    # Hover above the pre-grasp in joint space (the path there is not a straight line and must
    # stay clear of the item), then come down to the pre-grasp in a straight line.
    with sim.noisy(arm, rec):
        err = sim.move_to(arm, plan.pre + HOVER * Z, 1.6, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec).pos_err
    err = max(err, sim.move_line(arm, plan.pre, 0.5, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec))
    err = max(err, sim.move_line(arm, plan.grasp, 0.7, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec))
    disturbed = float(np.linalg.norm(sim.body_pos(item)[:2] - start[:2]))  # pushed while approaching
    sim.grip(arm, GRIPPER_CLOSED, 0.6, rec)
    sim.move_line(arm, sim.tip(arm) + 0.03 * Z, 0.3, point=sim.pointing(arm), on_step=rec)  # clear the table first
    # Straight-line carry with the pointing pinned (turning gradually to CARRY_TILT toward the
    # drop point): position + pointing use all 5 joints, so the posture can't drift, and a
    # joint-space carry across the table swung the item off target.
    dest = np.array([0, sg * 0.27, cfg["carry_z"]])
    t = np.deg2rad(CARRY_TILT)
    carry_point = np.cos(t) * DOWN + np.sin(t) * radial(sim, arm, dest)
    with sim.noisy(arm, rec):  # free-space part: lift and carry; fades out before the drop
        err = max(err, sim.move_line(arm, sim.tip(arm) + 0.09 * Z, 0.6, point=sim.pointing(arm), on_step=rec))
        err = max(err, sim.move_line(arm, dest, 1.6, point=carry_point, waypoints=30, on_step=rec))
    sim.step(150, rec)
    held = bool(sim.touching(arm, item)) and sim.body_pos(item)[2] > TABLE_Z + 0.05
    sim.grip(arm, cfg["open"], 0.5, rec)
    sim.move_line(arm, sim.tip(arm) + [0, -sg * 0.08, 0.04], 0.6, on_step=rec)
    go_home(sim, arm, rec)
    sim.step(500, rec)
    return dict(disturbed_cm=round(100 * disturbed, 1), held=held, ik_err_cm=round(100 * err, 1),
                in_drawer=inside(sim, f"{cfg['drawer']}_interior", sim.body_pos(item)))


def task1_episode(item: str, xy, seed: int = 0, rec=None, demo_noise: float = DEMO_NOISE) -> dict:
    """Task 1 with the expert: closer arm picks, the other opens the item's drawer first."""
    picker = "left" if xy[0] < 0 else "right"
    opener = "right" if picker == "left" else "left"
    drawer = ITEMS[item]["drawer"]
    sim = Sim(seed=seed)
    sim.demo_noise = demo_noise
    sim.clear_items()
    sim.place(item, (xy[0], xy[1], TABLE_Z + ITEMS[item]["half_h"] + 0.001))
    sim.step(50)
    row = dict(item=item, x=round(xy[0], 3), y=round(xy[1], 3), seed=seed, picker=picker)
    plan = plan_grasp(sim, picker, item)
    if plan is None:
        return row | dict(result="no grasp", success=False)
    o = open_drawer(sim, opener, drawer, rec)
    go_home(sim, opener, rec, via=drawer_via(opener, drawer))
    contacts_opening = sim.arm_contacts
    plan = plan_grasp(sim, picker, item)  # the item may have been nudged; replan from its current pose
    if plan is None:
        return row | dict(result="no grasp after opening", success=False)
    p = pick_and_place(sim, picker, item, plan, rec)
    success = p["held"] and p["in_drawer"] and sim.arm_contacts == 0 and o["opened"] >= 0.10
    return row | dict(result="ok", success=success, opened_cm=round(100 * o["opened"], 1), **p,
                      arm_contacts=sim.arm_contacts, contacts_opening=contacts_opening,
                      max_jump=round(sim.max_jump, 2))

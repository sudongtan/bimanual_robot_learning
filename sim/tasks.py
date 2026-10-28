"""Grasp planning and task geometry for the workcell (tasks S1-S4, T1, T2 in 1tasks_scene.md).

Grasps are fixed per target so demonstrations stay consistent:
    cube:   from above, tilted CUBE_TILT outward along the arm's reach direction, fingertips
            2 cm above the cube's bottom. Jaws close across the reach direction on the table,
            and along x (parallel to the drawer front) inside the drawer.
    handle: from above, tilted HANDLE_TILT, jaws closing along y (across the bar), so pulling
            or pushing along y presses a jaw against the bar
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.scene import DOWN, TABLE_Z, Sim, grasp_target, solve_ik

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

CUBE_W = 0.04  # m, cube footprint
CUBE_H = 0.04  # m, cube height
CUBE_OPEN = 1.0  # rad gripper opening for the cube on the table (~7.5 cm between fingertips)
CUBE_OPEN_DRAWER = 0.8  # rad, narrower inside the drawer (~6 cm) so the open jaw clears the side walls
CUBE_TILT = 20  # degrees from vertical
HANDLE_CLOSED = np.array([0.0, 0.025, 0.355])  # handle bar centre with the drawer shut
HANDLE_OPEN = 0.5  # rad gripper opening for the handle (~4 cm; bar is 1.4 cm)
HANDLE_TILT = 20  # degrees from vertical
DRAWER_OPEN = 0.14  # m, opening used by S1/S4 = the end stop
# Opening range when the drawer starts partly open with the cube on the table (S2, S3). Above
# ~13 cm the handle is close enough to the back of the cube's table area that the open gripper
# touches it (S3: 3 of 156 starts had no grasp at 13-14 cm).
DRAWER_PARTLY_OPEN = (0.06, 0.12)
DRAWER_FRONT_Y = 0.08  # m, cabinet front (y of the closed drawer front)
PRE_GRASP = 0.06  # m back from the grasp along the pointing direction
POINT_TOL = np.deg2rad(8)

# Where the cube may start (from the reach maps: every cell here is graspable by both arms
# with the fixed 20° grasp). The table area stays clear of the open drawer and handle
# (handle reaches y = -0.115 at 14 cm) and keeps the cube 3 cm inside the table edge (y = -0.25).
TABLE_AREA = ((-0.08, 0.08), (-0.20, -0.15))  # (x range, y range), world frame; the far corners at y = -0.21 failed
# Inside the drawer, in the drawer's closed-position frame (world y = y_closed - opening).
# At DRAWER_OPEN this is world x -0.01..0.01, y -0.015..0.01, inside the patch both arms reach.
DRAWER_AREA = ((-0.01, 0.01), (0.125, 0.15))  # failures clustered at the corners of ±0.02 x 0.12-0.16
DRAWER_FLOOR_Z = 0.326
CUBE_TARGETS = {"left": (-0.08, -0.18), "middle": (0.0, -0.18), "right": (0.08, -0.18)}  # named spots for S4/T2
S3_TARGET = CUBE_TARGETS["middle"]  # S3 always moves the cube here (many sources, one target; S4 is the reverse)
# A table start never lies within this of S3_TARGET, for S1 and S2 as well as S3, so the start
# distribution is the same for all three (a cube near the middle would otherwise mean "not S3").
# Above the 2 cm success radius, so no S3 start is already solved.
START_CLEARANCE = 0.03


def table_xy(rng: np.random.Generator) -> tuple[float, float]:
    """Cube start on the table: uniform over TABLE_AREA outside START_CLEARANCE of S3_TARGET."""
    (x0, x1), (y0, y1) = TABLE_AREA
    while True:
        xy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        if np.hypot(xy[0] - S3_TARGET[0], xy[1] - S3_TARGET[1]) >= START_CLEARANCE:
            return xy


def drawer_xy(rng: np.random.Generator, opening: float) -> tuple[float, float]:
    """Cube position (world x, y) inside the drawer at the given opening."""
    (x0, x1), (y0, y1) = DRAWER_AREA
    return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)) - opening


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


def place_cube_on_table(sim: Sim, xy):
    sim.place("cube", (xy[0], xy[1], TABLE_Z + CUBE_H / 2 + 0.001), quat=yaw_quat(sim.cube_yaw))


def place_cube_in_drawer(sim: Sim, xy):
    sim.place("cube", (xy[0], xy[1], DRAWER_FLOOR_Z + CUBE_H / 2 + 0.001), quat=yaw_quat(sim.cube_yaw_drawer))


@dataclass
class GraspPlan:
    point: np.ndarray
    jaw_perp: np.ndarray
    grasp: np.ndarray
    pre: np.ndarray
    q_grasp: np.ndarray
    q_pre: np.ndarray


def radial(sim: Sim, arm: str, p) -> np.ndarray:
    """Horizontal unit vector from the arm's pan axis toward p."""
    r = np.asarray(p, float) - sim.data.body(f"{arm}/shoulder").xpos
    r[2] = 0
    return r / np.linalg.norm(r)


def tilted_down(rad: np.ndarray, tilt_deg: float) -> np.ndarray:
    t = np.deg2rad(tilt_deg)
    return np.cos(t) * DOWN + np.sin(t) * rad


def handle_pos(opening: float) -> np.ndarray:
    return HANDLE_CLOSED - opening * Y


def _contacts(model, data, arm_geoms: set[int], bodies: set[int], geoms: set[int]) -> list[str]:
    hits = []
    for c in data.contact[: data.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a in arm_geoms and (model.geom_bodyid[b] in bodies or b in geoms):
                hits.append(model.geom(b).name or model.body(model.geom_bodyid[b]).name)
    return hits


def _check_poses(sim: Sim, arm: str, qs, grip: float, avoid_bodies: set[str]) -> list[str]:
    """Contacts of `arm` (gripper open at `grip`) with the given bodies or the other arm, per pose."""
    a = sim.arms[arm]
    other = sim.arms["right" if arm == "left" else "left"].geoms
    bodies = {sim.model.body(b).id for b in avoid_bodies}
    scratch = mujoco.MjData(sim.model)
    hits = []
    for q in qs:
        scratch.qpos[:] = sim.data.qpos
        scratch.qpos[a.qadr] = q
        scratch.qpos[a.grip_qadr] = grip
        mujoco.mj_forward(sim.model, scratch)
        hits += _contacts(sim.model, scratch, a.geoms, bodies, other)
    return hits


def cube_in_drawer(sim: Sim) -> bool:
    return bool(sim.body_pos("cube")[2] > TABLE_Z + CUBE_H / 2 + 0.005)


def cube_open(sim: Sim) -> float:
    """Gripper opening for grasping the cube where it is now."""
    return CUBE_OPEN_DRAWER if cube_in_drawer(sim) else CUBE_OPEN


def plan_cube_grasp(sim: Sim, arm: str, tilt: float = CUBE_TILT, grip_height: float = 0.02, jaw_perp=None) -> GraspPlan | None:
    """Fixed top grasp of the cube where it is now; None if out of reach or the open gripper
    would touch the cube, table, cabinet, drawer or the other arm at the grasp or pre-grasp.
    `grip_height`: fingertips this far above the cube's bottom. `jaw_perp`: override which face
    pair the jaws close on (the jaws close perpendicular to this axis)."""
    centre = sim.body_pos("cube")
    rad = radial(sim, arm, centre)
    # Faces are taken from the cube's own horizontal axes, so a rotated cube (random yaw) is
    # still gripped flat on two faces.
    R = sim.data.body("cube").xmat.reshape(3, 3)
    # In the drawer: jaws open along x, parallel to the drawer front; opening toward the
    # front/back walls would hit them. On the table: jaws close across the reach direction (on
    # the two faces most parallel to it), so the tilted approach slides the open jaws past the
    # cube's sides instead of into its near edge.
    ref = Y if cube_in_drawer(sim) else rad
    face = R[:, 0] if abs(R[:, 0] @ ref) > abs(R[:, 1] @ ref) else R[:, 1]
    if jaw_perp is not None:
        face = np.asarray(jaw_perp, float)
    c = centre + (grip_height - CUBE_H / 2) * Z
    return plan_top_grasp(sim, arm, c, face, CUBE_W, cube_open(sim), "cube", tilt, mid_roll=True)


def plan_top_grasp(sim: Sim, arm: str, c, face, width: float, grip_open: float, body: str, tilt: float = CUBE_TILT,
                   camera_away_from=None, mid_roll: bool = False) -> GraspPlan | None:
    """The fixed top grasp: point `tilt`° from vertical along the arm's reach direction, jaws
    closing perpendicular to `face` around point `c` on an object of `width`; pre-grasp
    PRE_GRASP back along the pointing. None if unreachable or the open gripper would touch
    `body`, the furniture or the other arm.

    `camera_away_from`: also try the wrist rolled 180° and keep the variant whose wrist-camera
    mount (it sticks out on one side) is farther from this point. Used in the hand-off, where
    the two mounts otherwise meet. Without it, the IK's first answer is used, as before.

    `mid_roll`: also try the wrist rolled 180° and keep the variant whose wrist roll is nearer the
    middle of its range. Used for the cube: a rotated cube can put the roll near its limit, and the
    roll then runs out while carrying (joint jumps on the far-side S3/S4 targets)."""
    a = sim.arms[arm]
    pt = tilted_down(radial(sim, arm, c), tilt)
    r0 = solve_ik(sim.model, a, c, point=pt, jaw_perp=face, restarts=8)
    if not r0.ok(point_tol=POINT_TOL):
        return None
    seeds = [r0.q]
    if camera_away_from is not None or mid_roll:
        q = r0.q.copy()
        q[4] += np.pi if q[4] < 0 else -np.pi
        # Clip rather than skip: a flip landing just past the roll limit still leads the IK to a
        # valid flipped posture.
        q[4] = np.clip(q[4], a.lo[4] + 0.05, a.hi[4] - 0.05)
        seeds.append(q)
    scratch = mujoco.MjData(sim.model)
    best = None
    for seed in seeds:
        r1 = solve_ik(sim.model, a, c, point=pt, jaw_perp=face, q0=seed)
        if not r1.ok(point_tol=POINT_TOL):
            continue
        scratch.qpos[:] = sim.data.qpos
        scratch.qpos[a.qadr] = r1.q
        mujoco.mj_kinematics(sim.model, scratch)
        g = grasp_target(c, scratch.site_xmat[a.site].reshape(3, 3)[:, 2], width)
        r2 = solve_ik(sim.model, a, g, point=pt, jaw_perp=face, q0=r1.q)
        # Along the pointing. (Straight down from above was tried: 195/240 S3 runs, failing on the
        # far side of the table area.)
        pre = g - PRE_GRASP * pt
        r3 = solve_ik(sim.model, a, pre, point=pt, jaw_perp=face, q0=r2.q, restarts=4)
        if not (r2.ok(point_tol=POINT_TOL) and r3.ok(point_tol=POINT_TOL)):
            continue
        if _check_poses(sim, arm, (r2.q, r3.q), grip_open, {body, "table", "cabinet", "drawer"}):
            continue
        plan = GraspPlan(point=pt, jaw_perp=face, grasp=g, pre=pre, q_grasp=r2.q, q_pre=r3.q)
        if camera_away_from is not None:
            scratch.qpos[a.qadr] = r2.q
            mujoco.mj_kinematics(sim.model, scratch)
            score = float(np.linalg.norm(scratch.geom(f"{arm}/camera_box2").xpos - np.asarray(camera_away_from)))
        elif mid_roll:
            score = -abs(r2.q[4] - (a.lo[4] + a.hi[4]) / 2)
        else:
            return plan
        if best is None or score > best[0]:
            best = (score, plan)
    return None if best is None else best[1]


# --- bar (T1) -------------------------------------------------------------------------
BAR_L, BAR_W = 0.12, 0.03  # m, length and width/height
BAR_GRIP_IN = 0.03  # m, each arm grips this far in from its own end
BAR_OPEN = 0.9  # rad gripper opening for the bar (~7 cm between fingertips)
BAR_AREA = ((-0.02, 0.02), (-0.20, -0.15))  # bar centre start area; bar lies along x
BAR_TARGETS = {"left": (-0.04, -0.175), "middle": (0.0, -0.175), "right": (0.04, -0.175)}


def place_bar_on_table(sim: Sim, xy, yaw: float = 0.0):
    sim.place("bar", (xy[0], xy[1], TABLE_Z + BAR_W / 2 + 0.001), quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))


def bar_grip_point(sim: Sim, arm: str) -> np.ndarray:
    """Point BAR_GRIP_IN from the bar's end nearer to `arm`, on the bar's axis."""
    R = sim.data.body("bar").xmat.reshape(3, 3)
    c = sim.body_pos("bar")
    axis = R[:, 0]
    base = sim.data.body(f"{arm}/shoulder").xpos
    sign = 1.0 if (base - c) @ axis > 0 else -1.0  # toward the arm's own end
    return c + sign * (BAR_L / 2 - BAR_GRIP_IN) * axis


def plan_bar_grasp(sim: Sim, arm: str, tilt: float = CUBE_TILT, camera_away_from=None) -> GraspPlan | None:
    """Top grasp near the arm's own end of the bar, jaws closing across the bar's width."""
    axis = sim.data.body("bar").xmat.reshape(3, 3)[:, 0]
    return plan_top_grasp(sim, arm, bar_grip_point(sim, arm), axis, BAR_W, BAR_OPEN, "bar", tilt, camera_away_from)


def plan_handle_grasp(sim: Sim, arm: str, opening: float | None = None, tilt: float = HANDLE_TILT) -> GraspPlan | None:
    """Fixed top grasp of the handle bar at the drawer's current (or given) opening."""
    a = sim.arms[arm]
    h = handle_pos(sim.drawer_opening() if opening is None else opening)
    pt = tilted_down(radial(sim, arm, h), tilt)
    r1 = solve_ik(sim.model, a, h, point=pt, jaw_perp=X, restarts=8)
    if not r1.ok(point_tol=POINT_TOL):
        return None
    scratch = mujoco.MjData(sim.model)
    scratch.qpos[:] = sim.data.qpos
    scratch.qpos[a.qadr] = r1.q
    mujoco.mj_kinematics(sim.model, scratch)
    g = grasp_target(h, scratch.site_xmat[a.site].reshape(3, 3)[:, 2], 0.014)
    r2 = solve_ik(sim.model, a, g, point=pt, jaw_perp=X, q0=r1.q)
    pre = g - PRE_GRASP * pt
    r3 = solve_ik(sim.model, a, pre, point=pt, jaw_perp=X, q0=r2.q, restarts=4)
    if not (r2.ok(point_tol=POINT_TOL) and r3.ok(point_tol=POINT_TOL)):
        return None
    if _check_poses(sim, arm, (r2.q, r3.q), HANDLE_OPEN, {"table", "cabinet", "drawer", "cube"}):
        return None
    return GraspPlan(point=pt, jaw_perp=X, grasp=g, pre=pre, q_grasp=r2.q, q_pre=r3.q)


# --- T1 hand-off ------------------------------------------------------------------------
MEET = np.array([0.0, -0.08, 0.40])  # in front of the table centre, off the line between the bases;
# low enough that the giver can point straight down (block upright)
HANDOFF_GIVER_GRIP = 0.028  # giver's fingertips 1.2 cm below the cube's top, leaving the lower half free
HANDOFF_TAKER_LOW = 0.012  # taker grips this far below the cube's centre
# The giver grips the block by its front/back faces (jaws along y), so the faces toward both arms
# (±x) stay free; the taker then grips the lower part of the same front/back faces from its side.
HANDOFF_GIVER_JAW_PERP = X


def plan_taker_grasp(sim: Sim, taker: str) -> GraspPlan | None:
    """Side grasp of the lower part of the held cube: jaws close across the two cube faces most
    parallel to the taker's reach direction (as on the table); pointing along that direction,
    tilted down as little as possible. None if no tilt works without touching the giver or the cube."""
    a = sim.arms[taker]
    R = sim.data.body("cube").xmat.reshape(3, 3)
    target = sim.body_pos("cube") - HANDOFF_TAKER_LOW * R[:, 2]
    rad = radial(sim, taker, target)
    # Jaw axis must lie along a face normal, not diagonally: constrain it perpendicular to the
    # cube's horizontal axis that points most along the reach direction.
    face = R[:, 0] if abs(R[:, 0] @ rad) > abs(R[:, 1] @ rad) else R[:, 1]
    for t in range(0, 91, 15):
        pt = np.cos(np.deg2rad(t)) * rad + np.sin(np.deg2rad(t)) * DOWN
        r0 = solve_ik(sim.model, a, target, point=pt, jaw_perp=face, restarts=8)
        if not r0.ok(point_tol=POINT_TOL):
            continue
        # Try both wrist rolls: the SO-101's fixed jaw is on one side, and which side faces the
        # giver decides whether the open gripper clears it.
        for flip in (0.0, np.pi, -np.pi):
            seed_q = r0.q.copy()
            seed_q[4] += flip
            if not a.lo[4] <= seed_q[4] <= a.hi[4]:
                continue
            plan = _taker_plan_from(sim, taker, target, pt, face, seed_q)
            if plan is not None:
                return plan
    return None


def _taker_plan_from(sim: Sim, taker: str, target, pt, face, seed_q) -> GraspPlan | None:
    a = sim.arms[taker]
    if True:
        r1 = solve_ik(sim.model, a, target, point=pt, jaw_perp=face, q0=seed_q)
        if not r1.ok(point_tol=POINT_TOL):
            return None
        scratch = mujoco.MjData(sim.model)
        scratch.qpos[:] = sim.data.qpos
        scratch.qpos[a.qadr] = r1.q
        mujoco.mj_kinematics(sim.model, scratch)
        g = grasp_target(target, scratch.site_xmat[a.site].reshape(3, 3)[:, 2], CUBE_W)
        r2 = solve_ik(sim.model, a, g, point=pt, jaw_perp=face, q0=r1.q)
        pre = g - PRE_GRASP * pt
        r3 = solve_ik(sim.model, a, pre, point=pt, jaw_perp=face, q0=r2.q, restarts=4)
        if not (r2.ok(point_tol=POINT_TOL) and r3.ok(point_tol=POINT_TOL)):
            return None
        if _check_poses(sim, taker, (r2.q, r3.q), CUBE_OPEN, {"cube", "table", "cabinet", "drawer"}):
            return None
        return GraspPlan(point=pt, jaw_perp=face, grasp=g, pre=pre, q_grasp=r2.q, q_pre=r3.q)

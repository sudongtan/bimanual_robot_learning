"""Hand-off (T1) set-up search: choose the meeting point, the giver's wrist pose and the taker's
grasp together, so the taker's whole approach is collision-free.

The block's pose at the meeting point follows from the giver's real grip, measured once from an
actual scripted pick (`grip_transform`). Candidates are scored by the smallest distance between
the taker (open gripper) and the giver + block along the approach. The chosen set-up is then used
unchanged for every episode, so demonstrations stay consistent.

    python -m sim.handoff        # run the search and print the best set-ups
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import tasks as T
from sim.scene import DOWN, Sim, grasp_target, load_model, solve_ik

GIVER_GRIP = T.HANDOFF_GIVER_GRIP
TAKER_LOW = T.HANDOFF_TAKER_LOW
STAGE = 0.04  # m behind the pre-grasp, reached in joint space before the straight approach


@dataclass(frozen=True)
class Setup:
    meet: tuple[float, float, float]  # giver fingertip position at the hand-off
    giver_tilt: float  # degrees from vertical
    giver_az: float  # degrees: direction of that tilt, relative to the giver's reach direction
    taker_az: float  # degrees: taker's approach direction relative to its reach direction
    taker_tilt: float  # degrees below horizontal
    taker_face: int  # 0 or 1: which cube axis the taker's jaw constraint is perpendicular to
    taker_flip: bool  # wrist rolled 180° from the IK's first answer


def rotz(v, deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def grip_transform(giver: str, xy=(0.0, -0.175)):
    """Block pose in the giver's gripperframe after a real scripted pick (no noise)."""
    from sim import skills as K

    sim = K.new_sim(0, 0.0)
    T.place_cube_on_table(sim, xy)
    assert K.pick_cube(sim, giver, grip_height=GIVER_GRIP, jaw_perp=T.HANDOFF_GIVER_JAW_PERP)
    sim.step(100)
    a = sim.arms[giver]
    Rs, ps = sim.data.site_xmat[a.site].reshape(3, 3), sim.data.site_xpos[a.site]
    Rc, pc = sim.data.body("cube").xmat.reshape(3, 3), sim.body_pos("cube")
    hold = float(sim.data.qpos[a.grip_qadr])  # gripper angle while holding the cube (stopped by it)
    return Rs.T @ (pc - ps), Rs.T @ Rc, hold


class Checker:
    """Scratch model with the giver holding the block at a pose; checks taker poses."""

    def __init__(self):
        self.sim = Sim(seed=None, noise=0)
        self.model = self.sim.model
        self.d = mujoco.MjData(self.model)
        self.base_qpos = self.sim.data.qpos.copy()
        self.bodies = {self.model.body(b).id for b in ("cube", "table", "cabinet", "drawer")}

    def contacts(self, taker: str, giver: str, with_bodies: bool = True) -> bool:
        tg, gg = self.sim.arms[taker].geoms, self.sim.arms[giver].geoms
        for c in self.d.contact[: self.d.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in tg and (b in gg or (with_bodies and self.model.geom_bodyid[b] in self.bodies)):
                    return True
        return False

    def clearance(self, taker: str, giver: str, distmax=0.05) -> float:
        """Smallest distance from the taker to the giver arm and the block."""
        fromto = np.zeros(6)
        others = list(self.sim.arms[giver].geoms) + [self.model.geom("cube").id]
        best = distmax
        for g1 in self.sim.arms[taker].geoms:
            for g2 in others:
                best = min(best, mujoco.mj_geomDistance(self.model, self.d, g1, g2, distmax, fromto))
        return best

    def clearance_arms(self, taker: str, giver: str, distmax=0.05) -> float:
        """Smallest distance from the taker to the giver arm only."""
        fromto = np.zeros(6)
        best = distmax
        for g1 in self.sim.arms[taker].geoms:
            for g2 in self.sim.arms[giver].geoms:
                best = min(best, mujoco.mj_geomDistance(self.model, self.d, g1, g2, distmax, fromto))
        return best

    def set_state(self, giver: str, qg, taker: str, qt, grip_t: float, cube_pose, grip_g: float):
        d = self.d
        d.qpos[:] = self.base_qpos
        a_g, a_t = self.sim.arms[giver], self.sim.arms[taker]
        d.qpos[a_g.qadr] = qg
        d.qpos[a_g.grip_qadr] = grip_g
        d.qpos[a_t.qadr] = qt
        d.qpos[a_t.grip_qadr] = grip_t
        j = self.model.body("cube").jntadr[0]
        adr = self.model.jnt_qposadr[j]
        d.qpos[adr : adr + 3] = cube_pose[0]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, cube_pose[1].reshape(-1))
        d.qpos[adr + 3 : adr + 7] = q
        mujoco.mj_forward(self.model, d)


def giver_pose(ck: Checker, giver: str, X, setup: Setup):
    """Giver joints at the meeting point and the resulting block pose; None if unreachable."""
    a = ck.sim.arms[giver]
    meet = np.array(setup.meet)
    h = rotz(T.radial(ck.sim, giver, meet), setup.giver_az)
    pt = np.cos(np.deg2rad(setup.giver_tilt)) * DOWN + np.sin(np.deg2rad(setup.giver_tilt)) * h
    r = solve_ik(ck.model, a, meet, point=pt, q0=ck.sim.home, restarts=8)  # pointing only; roll as IK leaves it
    if not r.ok(point_tol=np.deg2rad(5)):
        return None
    d = mujoco.MjData(ck.model)
    d.qpos[:] = ck.base_qpos
    d.qpos[a.qadr] = r.q
    mujoco.mj_kinematics(ck.model, d)
    Rs, ps = d.site_xmat[a.site].reshape(3, 3), d.site_xpos[a.site]
    return r.q, (ps + Rs @ X[0], Rs @ X[1])  # X = (cube offset, cube rotation, holding angle)


def taker_plan(ck: Checker, giver: str, taker: str, qg, cube_pose, setup: Setup):
    """Taker grasp, pre-grasp and staging joints for the setup; None if any is unreachable."""
    a = ck.sim.arms[taker]
    pc, Rc = cube_pose
    target = pc - TAKER_LOW * Rc[:, 2]
    h = rotz(T.radial(ck.sim, taker, target), setup.taker_az)
    t = np.deg2rad(setup.taker_tilt)
    pt = np.cos(t) * h + np.sin(t) * DOWN
    face = Rc[:, setup.taker_face]
    r0 = solve_ik(ck.model, a, target, point=pt, jaw_perp=face, q0=ck.sim.home, restarts=8)
    if not r0.ok(point_tol=T.POINT_TOL):
        return None
    seed = r0.q.copy()
    if setup.taker_flip:
        seed[4] += np.pi if seed[4] < 0 else -np.pi
        if not a.lo[4] <= seed[4] <= a.hi[4]:
            return None
    r1 = solve_ik(ck.model, a, target, point=pt, jaw_perp=face, q0=seed)
    if not r1.ok(point_tol=T.POINT_TOL):
        return None
    d = mujoco.MjData(ck.model)
    d.qpos[:] = ck.base_qpos
    d.qpos[a.qadr] = r1.q
    mujoco.mj_kinematics(ck.model, d)
    g = grasp_target(target, d.site_xmat[a.site].reshape(3, 3)[:, 2], T.CUBE_W)
    r2 = solve_ik(ck.model, a, g, point=pt, jaw_perp=face, q0=r1.q)
    r3 = solve_ik(ck.model, a, g - T.PRE_GRASP * pt, point=pt, jaw_perp=face, q0=r2.q)
    r4 = solve_ik(ck.model, a, g - (T.PRE_GRASP + STAGE) * pt, point=pt, jaw_perp=face, q0=r3.q)
    if not all(r.ok(point_tol=T.POINT_TOL) for r in (r2, r3, r4)):
        return None
    return dict(point=pt, jaw_perp=face, grasp=g, q_grasp=r2.q, q_pre=r3.q, q_stage=r4.q)


def evaluate(giver: str, taker: str, X, setup: Setup, ck: Checker | None = None):
    """Smallest clearance (m) along the taker's approach: home -> stage (joint space) -> pre ->
    grasp (straight), open gripper. None if unreachable or anything touches."""
    ck = ck or Checker()
    gp = giver_pose(ck, giver, X, setup)
    if gp is None:
        return None
    qg, cube_pose = gp
    tp = taker_plan(ck, giver, taker, qg, cube_pose, setup)
    if tp is None:
        return None
    path = [ck.sim.home + s * (tp["q_stage"] - ck.sim.home) for s in np.linspace(0, 1, 15)]
    path += [tp["q_stage"] + s * (tp["q_pre"] - tp["q_stage"]) for s in np.linspace(0, 1, 5)]
    path += [tp["q_pre"] + s * (tp["q_grasp"] - tp["q_pre"]) for s in np.linspace(0, 1, 8)]
    hold = X[2]
    clear = 1.0
    for q in path:
        ck.set_state(giver, qg, taker, q, T.CUBE_OPEN, cube_pose, hold)
        if ck.contacts(taker, giver):
            return None
    for q in (tp["q_stage"], tp["q_pre"], tp["q_grasp"]):
        ck.set_state(giver, qg, taker, q, T.CUBE_OPEN, cube_pose, hold)
        clear = min(clear, ck.clearance(taker, giver))
    # Closing on the cube: the taker's jaws stop at about the giver's holding angle; they may
    # touch the cube but not the giver.
    for grip in np.linspace(T.CUBE_OPEN, hold, 5):
        ck.set_state(giver, qg, taker, tp["q_grasp"], grip, cube_pose, hold)
        if ck.contacts(taker, giver, with_bodies=False):
            return None
        clear = min(clear, ck.clearance_arms(taker, giver))
    return clear


GRID = dict(
    meet=[(x, y, z) for x in (-0.03, 0.0, 0.03) for y in (-0.12, -0.08, -0.04) for z in (0.37, 0.40, 0.43)],
    giver_tilt=(0, 15, 30),
    giver_az=(0, 90, -90, 180),
    taker_az=(-40, -20, 0, 20, 40),
    taker_tilt=(0, 20, 40, 60),
    taker_face=(0, 1),
    taker_flip=(False, True),
)


def _search_giver(args):
    giver, taker, X, meet, gt, gaz = args
    ck = Checker()
    out = []
    base = Setup(meet, gt, gaz, 0, 0, 0, False)
    if giver_pose(ck, giver, X, base) is None:
        return out
    for taz, tt, tf, fl in itertools.product(GRID["taker_az"], GRID["taker_tilt"], GRID["taker_face"], GRID["taker_flip"]):
        s = Setup(meet, gt, gaz, taz, tt, tf, fl)
        c = evaluate(giver, taker, X, s, ck)
        if c is not None:
            out.append((c, s))
    return out


def search(giver: str = "left", taker: str = "right", top: int = 10):
    X = grip_transform(giver)
    jobs = [(giver, taker, X, m, gt, gaz) for m in GRID["meet"] for gt in GRID["giver_tilt"] for gaz in GRID["giver_az"]]
    with ProcessPoolExecutor() as pool:
        found = [r for chunk in pool.map(_search_giver, jobs) for r in chunk]
    found.sort(key=lambda r: -r[0])
    return found[:top], len(found)


if __name__ == "__main__":
    giver = sys.argv[1] if len(sys.argv) > 1 else "left"
    taker = "right" if giver == "left" else "left"
    best, n = search(giver, taker)
    print(f"giver {giver} -> taker {taker}: {n} collision-free set-ups")
    for c, s in best:
        print(f"  clearance {c * 1000:5.1f} mm  {s}")

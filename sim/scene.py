"""Workcell scene: model loading, per-arm bookkeeping, IK and a scripted controller.

Frame: table centre at x = y = 0, +x toward arm B (right), +y toward the back (cabinet),
tabletop at z = 0.315 m. Arm A = "left", arm B = "right".

Gripper frame (`<arm>/gripperframe` site, between the fingertips):
    x-axis = pointing direction, z-axis = jaw closing direction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SCENE_XML = Path(__file__).parent / "scenes" / "workcell.xml"

TABLE_Z = 0.315
ARMS = ("left", "right")
ARM_NAME = {"A": "left", "B": "right"}  # names used in commands
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DRAWER_PULL = 0.14  # m, full travel of the drawer joint (end stop)
ITEMS = ("cube",)
DISTRACTORS = ("puck", "cylinder", "pebble")

GRIPPER_OPEN = 1.2  # rad, ~9 cm between fingertips
GRIPPER_CLOSED = -0.17  # rad, fully shut; a held object stops the jaw earlier

# Home: upright over the arm's own base, gripper pointing toward the other arm and ~35° up;
# every joint >= 0.23 rad inside its limits. The right arm is the left arm rotated 180°, so
# it uses the same joint angles. (The earlier diagonal home, fingertip at (-0.072, 0.117,
# 0.449), leaned each arm over the drawer the other arm delivers to and caused collisions.)
HOME_Q = np.array([-0.04, -1.51, -0.42, 1.34, 0.11])
HOME_TIP_LEFT = np.array([-0.168, 0.0, 0.689])  # fingertip at HOME_Q, for reference
START_NOISE = 0.02  # rad, σ per arm joint at reset
DEMO_NOISE = 0.008  # rad, σ of the demonstration noise, as validated in the deprecated expert (fingertip ~3 mm RMS)
NOISE_TAU = 0.25  # s, correlation time of demonstration noise (smooth drift, not jitter)
NOISE_RAMP = 0.2  # s, demonstration noise fades in/out over this time
CONTROL_EVERY = 10  # physics steps per command for recorded episodes: 50 Hz zero-order hold, one command per
# 50 fps frame, as ACT's scripted sim policy (2data_collection.md, design point 1)
# What Sim.step saves before each tick: the full physics state plus the solver's warm start, so replaying the
# recorded commands from it is exact. Not ctrl: that is the recorded action.
STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_WARMSTART

DOWN = np.array([0.0, 0.0, -1.0])
LIMIT_MARGIN = 0.05  # rad; IK stays this far inside joint limits (a joint at its limit gets pushed back)

# The SO-101 gripper is asymmetric: the fixed jaw's inner face sits 2 cm from the
# gripperframe along its z-axis and only the other jaw moves. Aiming the gripperframe
# at an item's centre lands the fixed jaw on anything wider than ~4 cm.
FIXED_JAW_Z = -0.020
JAW_CLEARANCE = 0.005


def grasp_target(centre, jaw_axis, width: float) -> np.ndarray:
    """Gripperframe position that leaves the fixed jaw JAW_CLEARANCE outside an item of `width`."""
    return np.asarray(centre) + (-FIXED_JAW_Z - width / 2 - JAW_CLEARANCE) * np.asarray(jaw_axis)


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


@dataclass
class Arm:
    """Indices for one SO-101 arm inside the combined scene model."""

    name: str
    qadr: np.ndarray  # qpos addresses of the 5 arm joints
    dadr: np.ndarray  # dof addresses of the 5 arm joints
    lo: np.ndarray
    hi: np.ndarray
    act: np.ndarray  # actuator ids of the 5 arm joints
    grip_act: int
    grip_qadr: int
    site: int  # gripperframe
    shoulder_body: int
    geoms: set[int] = field(default_factory=set)  # collision geoms of this arm

    @classmethod
    def from_model(cls, model: mujoco.MjModel, name: str) -> Arm:
        joints = [model.joint(f"{name}/{j}") for j in ARM_JOINTS]
        bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(f"{name}/")}
        geoms = {
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
        }
        return cls(
            name=name,
            qadr=np.array([j.qposadr[0] for j in joints]),
            dadr=np.array([j.dofadr[0] for j in joints]),
            lo=np.array([j.range[0] for j in joints]),
            hi=np.array([j.range[1] for j in joints]),
            act=np.array([model.actuator(f"{name}/{j}").id for j in ARM_JOINTS]),
            grip_act=model.actuator(f"{name}/gripper").id,
            grip_qadr=model.joint(f"{name}/gripper").qposadr[0],
            site=model.site(f"{name}/gripperframe").id,
            shoulder_body=model.body(f"{name}/shoulder").id,
            geoms=geoms,
        )


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float  # m
    point_err: float  # rad between gripper x-axis and the requested pointing direction (0 if none)
    jaw_err: float  # |jaw · bar|, 0 when the jaws close across the bar (0 if none)

    def ok(self, pos_tol=0.005, point_tol=np.deg2rad(15), jaw_tol=0.2) -> bool:
        return self.pos_err < pos_tol and self.point_err < point_tol and self.jaw_err < jaw_tol


def solve_ik(
    model: mujoco.MjModel,
    arm: Arm,
    pos,
    point=None,
    jaw_perp=None,
    q0=None,
    restarts: int = 0,
    seed: int = 0,
    iters: int = 150,
) -> IKResult:
    """Damped least squares IK for one arm.

    pos: fingertip (gripperframe) target. point: desired pointing direction (gripper x-axis).
    jaw_perp: a direction the jaws must close across, e.g. a handle bar's axis.
    Tries q0 first, then `restarts` random starts. Returns the successful solution
    closest to q0, or the least-bad one if none succeeds.
    """
    data = mujoco.MjData(model)
    pos = np.asarray(pos, float)
    rng = np.random.default_rng(seed)
    home = np.zeros(5) if q0 is None else np.asarray(q0, float)
    starts = [home] + [rng.uniform(arm.lo, arm.hi) for _ in range(restarts)]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    best, good = None, []
    for q in starts:
        q = q.copy()
        for _ in range(iters):
            data.qpos[arm.qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jacp, jacr, arm.site)
            R = data.site_xmat[arm.site].reshape(3, 3)
            rows, errs = [jacp[:, arm.dadr]], [pos - data.site_xpos[arm.site]]
            if point is not None:
                # Rotation that turns x onto `point` is ~ x × point; weight 0.1 m per rad.
                rows.append(0.1 * jacr[:, arm.dadr])
                errs.append(0.1 * np.cross(R[:, 0], point))
            if jaw_perp is not None:
                # d(jaw · b)/dt = ω · (jaw × b)
                rows.append(0.1 * (np.cross(R[:, 2], jaw_perp) @ jacr[:, arm.dadr])[None])
                errs.append(0.1 * np.array([-R[:, 2] @ jaw_perp]))
            J, e = np.vstack(rows), np.concatenate(errs)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), arm.lo + LIMIT_MARGIN, arm.hi - LIMIT_MARGIN)
        data.qpos[arm.qadr] = q
        mujoco.mj_kinematics(model, data)
        R = data.site_xmat[arm.site].reshape(3, 3)
        res = IKResult(
            q=q,
            pos_err=float(np.linalg.norm(pos - data.site_xpos[arm.site])),
            point_err=0.0 if point is None else float(np.arccos(np.clip(R[:, 0] @ point, -1, 1))),
            jaw_err=0.0 if jaw_perp is None else float(abs(R[:, 2] @ jaw_perp)),
        )
        score = res.pos_err + 0.05 * res.point_err + 0.05 * res.jaw_err
        if best is None or score < best[0]:
            best = (score, res)
        if res.ok():
            good.append(res)
            if q is home:
                break
    if good:
        return min(good, key=lambda r: np.linalg.norm(r.q - home))
    return best[1]


def home_q(model: mujoco.MjModel) -> np.ndarray:
    return HOME_Q.copy()


class Sim:
    """Scene plus a scripted joint-space controller that records arm-to-arm contacts."""

    def __init__(self, seed: int | None = 0, noise: float = START_NOISE):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.arms = {a: Arm.from_model(self.model, a) for a in ARMS}
        self.home = home_q(self.model)
        self.arm_contacts = 0  # simulation steps with any left–right contact
        self.demo_noise = 0.0  # rad, σ of the smooth offset added to arm commands inside noisy()
        self.cube_yaw = 0.0  # rad, cube rotation when the tasks place it on the table (sim/randomize.py sets it)
        self.cube_yaw_drawer = 0.0  # rad, the same inside the drawer (a smaller range)
        self.distractor_xy: dict[str, np.ndarray] = {}  # distractors on the table and where they started
        self.pushed = 0.0  # m, farthest an item moved while being approached for a grasp (sim/skills.py)
        # Control clock: the expert writes data.ctrl every step, but the actuators get it only every
        # control_every steps and hold it in between (1 = every step). At each such tick, before the
        # physics step, step() saves what a recorder needs:
        #   ticked        True during the on_step call of a tick step
        #   tick_state    physics state before the tick (STATE_SPEC)
        #   applied_ctrl  the command held until the next tick, with demonstration noise
        #   nominal_ctrl  the same without demonstration noise
        self.control_every = 1
        self.subtask = 0  # 0, then 1 once a composite's first arm is back home (sim/skills.py)
        self.reset(seed, noise)

    # --- state -------------------------------------------------------------
    def reset(self, seed: int | None = 0, noise: float = START_NOISE):
        mujoco.mj_resetData(self.model, self.data)
        self.rng = rng = np.random.default_rng(seed)
        self._offset = {a: np.zeros(5) for a in self.arms}  # demonstration-noise state per arm
        self._gate = {a: 0.0 for a in self.arms}
        self._gate_on = {a: False for a in self.arms}
        for arm in self.arms.values():
            q = np.clip(self.home + rng.normal(0, noise, 5), arm.lo, arm.hi)
            self.data.qpos[arm.qadr] = q
            self.data.ctrl[arm.act] = q
            self.data.qpos[arm.grip_qadr] = GRIPPER_CLOSED
            self.data.ctrl[arm.grip_act] = GRIPPER_CLOSED
        self.arm_contacts = 0
        self.step_count = 0  # physics steps since reset; a tick is every control_every-th step
        self.ticked = False
        self.subtask = 0
        self.max_jump = 0.0  # largest joint change between move_line waypoints (rad); big = IK switched branch
        mujoco.mj_forward(self.model, self.data)

    def place(self, body: str, pos, quat=(1, 0, 0, 0)):
        """Teleport a free body (item) and zero its velocity."""
        j = self.model.body(body).jntadr[0]
        a, v = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = quat
        self.data.qvel[v : v + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def clear_items(self):
        """Park every item on the floor away from the table (for tests that need an empty table)."""
        for i, body in enumerate(ITEMS):
            self.place(body, (1.0 + 0.3 * i, 1.0, 0.06))
        for i, body in enumerate(DISTRACTORS):
            self.place(body, (1.0 + 0.3 * i, 1.4, 0.06))
        self.distractor_xy = {}

    def pointing(self, arm: str) -> np.ndarray:
        """Gripper pointing direction (gripperframe x-axis)."""
        return self.data.site_xmat[self.arms[arm].site].reshape(3, 3)[:, 0].copy()

    def jaw_axis(self, arm: str) -> np.ndarray:
        """Jaw closing axis (gripperframe z-axis), pointing from the fixed jaw toward the moving one."""
        return self.data.site_xmat[self.arms[arm].site].reshape(3, 3)[:, 2].copy()

    def tip(self, arm: str) -> np.ndarray:
        return self.data.site_xpos[self.arms[arm].site].copy()

    def q(self, arm: str) -> np.ndarray:
        return self.data.qpos[self.arms[arm].qadr].copy()

    def drawer_opening(self) -> float:
        return float(self.data.joint("drawer").qpos[0])

    def set_drawer(self, opening: float):
        self.data.joint("drawer").qpos[0] = opening
        self.data.joint("drawer").qvel[0] = 0
        mujoco.mj_forward(self.model, self.data)

    def body_pos(self, body: str) -> np.ndarray:
        return self.data.body(body).xpos.copy()

    # --- contacts ----------------------------------------------------------
    def arms_touching(self) -> bool:
        lg, rg = self.arms["left"].geoms, self.arms["right"].geoms
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in lg and c.geom2 in rg) or (c.geom1 in rg and c.geom2 in lg):
                return True
        return False

    def touching(self, arm: str, body: str) -> set[int]:
        """Geom ids of `arm` currently in contact with `body`."""
        bid = self.model.body(body).id
        hits = set()
        for c in self.data.contact[: self.data.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.arms[arm].geoms and self.model.geom_bodyid[b] == bid:
                    hits.add(a)
        return hits

    def arm_clearance(self, distmax: float = 0.2) -> float:
        """Smallest distance between any left and any right collision geom (negative = overlap)."""
        best = distmax
        fromto = np.zeros(6)
        for g1 in self.arms["left"].geoms:
            for g2 in self.arms["right"].geoms:
                best = min(best, mujoco.mj_geomDistance(self.model, self.data, g1, g2, distmax, fromto))
        return best

    # --- motion ------------------------------------------------------------
    def step(self, n: int = 1, on_step=None):
        for _ in range(n):
            self.ticked = self.step_count % self.control_every == 0
            if self.ticked:
                self.tick_state = np.empty(mujoco.mj_stateSize(self.model, STATE_SPEC))
                mujoco.mj_getState(self.model, self.data, self.tick_state, STATE_SPEC)
                self.nominal_ctrl = self.data.ctrl.copy()
                self.applied_ctrl = self.nominal_ctrl + self._demo_noise_offset()
            expert = self.data.ctrl.copy()  # the expert reads its own (clean, latest) command back from ctrl
            self.data.ctrl[:] = self.applied_ctrl
            mujoco.mj_step(self.model, self.data)
            self.data.ctrl[:] = expert
            self.step_count += 1
            if self.arms_touching():
                self.arm_contacts += 1
            if on_step is not None:
                on_step(self)

    def _demo_noise_offset(self) -> np.ndarray:
        """Advance each arm's noise by one control interval (Ornstein-Uhlenbeck, so it drifts smoothly)
        and return the offset to add to ctrl: the noise on each arm's joints, scaled by a gate that fades in/out."""
        offset = np.zeros(self.model.nu)
        if self.demo_noise <= 0:
            return offset
        dt = self.control_every * self.model.opt.timestep
        for name, arm in self.arms.items():
            x = self._offset[name]
            x += -x * dt / NOISE_TAU + self.demo_noise * np.sqrt(2 * dt / NOISE_TAU) * self.rng.standard_normal(5)
            step = dt / NOISE_RAMP
            g = self._gate[name]
            self._gate[name] = min(1.0, g + step) if self._gate_on[name] else max(0.0, g - step)
            offset[arm.act] = self._gate[name] * x
        return offset

    @contextmanager
    def noisy(self, arm: str, on_step=None):
        """Demonstration noise on `arm` for the moves inside the block (free-space moves only).
        On exit the noise fades out and the arm settles, so the next (precise) move starts on target."""
        self._gate_on[arm] = True
        try:
            yield
        finally:
            self._gate_on[arm] = False
            if self.demo_noise > 0:
                while self._gate[arm] > 0:
                    self.step(1, on_step)
                self.step(int(0.1 / self.model.opt.timestep), on_step)

    def move_joints(self, targets: dict[str, np.ndarray], duration: float, grip: dict[str, float] | None = None, on_step=None):
        """Linearly interpolate arm joint targets over `duration` seconds; set gripper targets at the start."""
        grip = grip or {}
        for a, g in grip.items():
            self.data.ctrl[self.arms[a].grip_act] = g
        start = {a: self.data.ctrl[self.arms[a].act].copy() for a in targets}
        n = max(1, int(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep: zero velocity at both ends
            for a, q in targets.items():
                self.data.ctrl[self.arms[a].act] = start[a] + s * (np.asarray(q) - start[a])
            self.step(1, on_step)

    def move_line(self, arm: str, goal, duration: float, point=None, jaw_perp=None, waypoints: int = 20, on_step=None) -> float:
        """Move the fingertip along a straight line (IK per waypoint). Returns the worst IK position error."""
        a = self.arms[arm]
        start, start_point = self.tip(arm), self.pointing(arm)
        q = self.data.ctrl[a.act].copy()
        worst = 0.0
        for k in range(1, waypoints + 1):
            s = k / waypoints
            p = start + (np.asarray(goal) - start) * s
            # Turn the gripper gradually so each IK step starts close to its answer.
            pt = None if point is None else start_point + s * (np.asarray(point) - start_point)
            pt = None if pt is None else pt / np.linalg.norm(pt)
            # Warm start only: random restarts can land on another arm posture (e.g. the wrist
            # flipped 180°), which is a jump mid-path. Use move_to for large reorientations.
            res = solve_ik(self.model, a, p, point=pt, jaw_perp=jaw_perp, q0=q)
            if not res.ok():
                res = solve_ik(self.model, a, p, point=pt, jaw_perp=jaw_perp, q0=q, iters=500)
            self.max_jump = max(self.max_jump, float(np.abs(res.q - q).max()))
            q, worst = res.q, max(worst, res.pos_err)
            self.move_joints({arm: q}, duration / waypoints, on_step=on_step)
        return worst

    def move_to(self, arm: str, pos, duration: float, point=None, jaw_perp=None, on_step=None) -> IKResult:
        """Joint-space move to the IK solution closest to the current joints (restarts allowed),
        for approaches from far away or with a large change of gripper orientation."""
        a = self.arms[arm]
        res = solve_ik(self.model, a, pos, point=point, jaw_perp=jaw_perp, q0=self.data.ctrl[a.act].copy(), restarts=30)
        self.move_joints({arm: res.q}, duration, on_step=on_step)
        return res

    def grip(self, arm: str, value: float, duration: float = 0.5, on_step=None, ramp: bool = False):
        """Set the gripper target and wait `duration`; with `ramp`, move the target there gradually over `duration`."""
        act = self.arms[arm].grip_act
        n = int(duration / self.model.opt.timestep)
        if not ramp:
            self.data.ctrl[act] = value
            self.step(n, on_step)
            return
        start = float(self.data.ctrl[act])
        for i in range(1, n + 1):
            self.data.ctrl[act] = start + (value - start) * i / n
            self.step(1, on_step)

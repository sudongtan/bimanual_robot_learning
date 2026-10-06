"""Scripted experts: open-loop plans built at reset from privileged sim state.

An expert turns the task into joint-space waypoints per arm (solved with
`sim.ik.ArmIK`), then plays them back: each segment moves from the previous
waypoint to the next with a smooth (cosine) profile. It returns the full
12-value action the environment expects, every control step.
"""

import mujoco
import numpy as np

from sim.ik import ArmIK
from sim.waste_env import CONTROL_HZ, TABLE_TOP, WasteSortEnv

DOWN = np.array([0.0, 0.0, -1.0])
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.17    # rad; ~9.2 cm and ~0.6 cm fingertip gap
TIP = (0.012, 0.0, 0.003)              # between the fingertips, gripper-site frame
CAN_AXIS = (-0.017, 0.0, 0.0)          # can axis when held: 3 mm clear of the fixed jaw
CAN_H = 0.102                          # tomato soup can height (scan)
GRASP_Z = 0.37                         # fingertip height when grasping: 4.7 cm below the can's top
SLIDE = 0.085                          # start the slide-in this far beside the can's axis
PARK_B = (0.26, -0.02, 0.45)           # arm B's fingertip while arm A drops the can


class IKFailed(RuntimeError):
    pass


class Plan:
    """Per-arm waypoint lists: (duration s, 5 arm joints, gripper)."""

    def __init__(self, home_ctrl: np.ndarray):
        self.home = home_ctrl.copy()
        self.segments = {"left": [], "right": []}
        self.last = {"left": (home_ctrl[:5].copy(), home_ctrl[5]), "right": (home_ctrl[6:11].copy(), home_ctrl[11])}
        self.clock = {"left": 0.0, "right": 0.0}

    def move(self, side, q, grip, seconds, label=""):
        self.segments[side].append((self.clock[side], seconds, self.last[side], (np.asarray(q), grip), label))
        self.clock[side] += seconds
        self.last[side] = (np.asarray(q), grip)

    def wait_for(self, side, other):
        """Hold `side` until `other` has finished its moves so far."""
        if self.clock[side] < self.clock[other]:
            self.move(side, *self.last[side], self.clock[other] - self.clock[side], "wait")

    @property
    def duration(self) -> float:
        return max(self.clock.values())

    def action(self, t: float) -> np.ndarray:
        out = self.home.copy()
        for side, base in (("left", 0), ("right", 6)):
            q, g = self.last[side]
            for start, dur, (q0, g0), (q1, g1), _ in self.segments[side]:
                if t < start + dur:
                    s = np.clip((t - start) / dur, 0, 1)
                    s = 0.5 - 0.5 * np.cos(np.pi * s)
                    q, g = q0 + s * (q1 - q0), g0 + s * (g1 - g0)
                    break
            out[base:base + 5], out[base + 5] = q, g
        return out

    def phase(self, t: float) -> str:
        labels = []
        for side in ("left", "right"):
            for start, dur, _, _, label in self.segments[side]:
                if start <= t < start + dur and label != "wait":
                    labels.append(f"{side}: {label}")
        return ", ".join(labels) or "done"


class CanToMetalBinExpert:
    """Arm A (left) lifts the can, arm B (right) opens the metal bin, arm A drops the can in."""

    # Declared per expert so task-agnostic tools (smoke test, viewer) can check that the
    # grasped item was not disturbed before the grasp, without knowing the task.
    grasped_item = "tomato_soup_can"
    grasp_phase = "left: close"

    def __init__(self, env: WasteSortEnv):
        self.env = env
        self.ik = {}

    def _ik_for_current_model(self):
        """The environment builds one model per item set; keep the IK solvers on the current one."""
        if not self.ik or self.ik["left"].m is not self.env.model:
            self.ik = {"left": ArmIK(self.env.model, "left"), "right": ArmIK(self.env.model, "right")}

    def _solve(self, side, qpos, prev, target, offset, direction=None):
        """IK from the previous waypoint; fall back to restarts, keeping the solution nearest `prev`."""
        ik = self.ik[side]
        seed = qpos.copy()
        seed[ik.qadr] = prev
        q, ep, er = ik.solve(seed, target, offset, direction)
        if ep < 2e-3 and er < np.radians(3):
            return q
        rng = np.random.default_rng(0)
        best = None
        for _ in range(30):
            seed[ik.qadr] = rng.uniform(ik.lo, ik.hi)
            q, ep, er = ik.solve(seed, target, offset, direction)
            if ep < 2e-3 and er < np.radians(3):
                dist = np.abs(q - prev).sum()
                if best is None or dist < best[0]:
                    best = (dist, q)
        if best is None:
            raise IKFailed(f"{side} arm cannot reach {np.round(target, 3)}")
        return best[1]

    def _site_axes(self, side, qpos, q):
        """Gripper-site rotation matrix (columns = x, y, z axes) with the arm at `q`."""
        self._point(side, qpos, q, (0, 0, 0))
        return self.ik[side].d.site_xmat[self.ik[side].site].reshape(3, 3).copy()

    def _point(self, side, qpos, q, offset):
        """World position of `offset` (gripper-site frame) with the arm at joint angles `q`."""
        ik = self.ik[side]
        ik.d.qpos[:] = qpos
        ik.d.qpos[ik.qadr] = q
        mujoco.mj_kinematics(ik.m, ik.d)
        return ik.point(offset)

    def reset(self) -> Plan:
        self._ik_for_current_model()
        env, d = self.env, self.env.data
        qpos = d.qpos.copy()
        plan = Plan(env.model.key_ctrl[env.home_key])
        A, B = "left", "right"
        qa, qb = plan.last[A][0], plan.last[B][0]

        # --- arm A: grasp and lift the can --------------------------------------------------
        # The arm cannot hold the gripper vertical above the can's top (vertical reach ends at
        # z ~0.40, the can top is 0.417), so it lowers the open gripper beside the can and slides
        # it in sideways: the jaws are thin plates, so the gap between them is open at the sides.
        can = env.item_pos(self.grasped_item)
        grasp_pt = np.array([can[0], can[1], GRASP_Z])
        grasp = self._solve(A, qpos, qa, grasp_pt, CAN_AXIS, DOWN)
        slide = self._site_axes(A, qpos, grasp)[:, 1] * [1, 1, 0]
        slide /= np.linalg.norm(slide)
        for sign in (-1, 1):
            try:
                beside = self._solve(A, qpos, grasp, grasp_pt + sign * SLIDE * slide, CAN_AXIS, DOWN)
                hover = self._solve(A, qpos, beside, grasp_pt + sign * SLIDE * slide + [0, 0, 0.03], CAN_AXIS, DOWN)
                break
            except IKFailed:
                if sign == 1:
                    raise
        # Clear the can first, gripper closed: at home the jaws hang just above the can's top, and
        # opening them on the way swept the moving jaw through the can (seed 25 knocked it over).
        hover_pt = grasp_pt + sign * SLIDE * slide + [0, 0, 0.03]
        clear = self._solve(A, qpos, hover, hover_pt + [0, 0, 0.08], TIP)
        plan.move(A, clear, GRIP_CLOSED, 1.0, "up and over, closed")
        plan.move(A, hover, GRIP_CLOSED, 0.8, "beside the can")
        plan.move(A, hover, GRIP_OPEN, 0.4, "open")
        plan.move(A, beside, GRIP_OPEN, 0.6, "down beside the can")
        plan.move(A, grasp, GRIP_OPEN, 1.0, "slide the jaws around the can")
        plan.move(A, grasp, GRIP_CLOSED, 0.6, "close")
        # The can's centre in the gripper-site frame, as held: carry targets move this point.
        held = self._site_axes(A, qpos, grasp).T @ (can - self._point(A, qpos, grasp, (0, 0, 0)))
        lift = self._solve(A, qpos, grasp, can + [0, 0, 0.10], held)
        plan.move(A, lift, GRIP_CLOSED, 1.0, "lift")

        # --- arm B: hook the metal-bin handle and pull it open --------------------------------
        plan.wait_for(B, A)
        hook = d.site("metal_bin_hook").xpos.copy()
        q = self._solve(B, qpos, qb, hook + [0, 0, 0.06], TIP)
        plan.move(B, q, GRIP_CLOSED, 1.5, "above the handle")
        q = self._solve(B, qpos, q, hook, TIP)
        plan.move(B, q, GRIP_CLOSED, 0.8, "into the gap")
        pull = 0.20                                   # past the 0.18 m stop: the fingertip lags the target
        for s in np.linspace(0.03, pull, 6):
            q = self._solve(B, qpos, q, hook + [0, -s, 0], TIP)
            plan.move(B, q, GRIP_CLOSED, 0.4, "pull the bin open")
        # Leave the gap upward and slightly outward, so the fingertip does not push the bin back in.
        q = self._solve(B, qpos, q, hook + [0, -pull - 0.01, 0.08], TIP)
        plan.move(B, q, GRIP_CLOSED, 0.8, "out of the gap")
        # Park beside its own base, not at home: at home arm B's wrist camera is in arm A's drop path.
        q = self._solve(B, qpos, q, np.array(PARK_B), TIP)
        plan.move(B, q, GRIP_CLOSED, 1.5, "out of arm A's way")

        # --- arm A: carry the can over the open bin and drop it ------------------------------
        plan.wait_for(A, B)
        # Bin interior centre once open: closed centre moved 0.18 m toward -y.
        inside = d.site("metal_bin_interior").xpos.copy() + [0, -0.18, 0]
        q = self._solve(A, qpos, lift, np.array([inside[0], inside[1] + 0.06, TABLE_TOP + 0.16]), held)
        plan.move(A, q, GRIP_CLOSED, 1.8, "carry toward the bin")
        q = self._solve(A, qpos, q, np.array([inside[0], inside[1], TABLE_TOP + 0.05]), held)
        plan.move(A, q, GRIP_CLOSED, 1.0, "over the open bin")
        plan.move(A, q, GRIP_OPEN, 0.5, "release")
        up = self._solve(A, qpos, q, self._point(A, qpos, q, TIP) + [0, 0, 0.06], TIP)
        plan.move(A, up, GRIP_OPEN, 0.7, "up")
        plan.move(A, plan.home[:5], plan.home[5], 1.5, "back to home")
        self.plan = plan
        return plan

    def act(self, steps: int) -> np.ndarray:
        return self.plan.action(steps / CONTROL_HZ)


# One scripted expert per task. Every expert has: reset() -> Plan, act(step) -> 12-value action,
# and the attributes `grasped_item` and `grasp_phase` (see CanToMetalBinExpert).
EXPERTS = {
    "can_to_metal_bin": CanToMetalBinExpert,
}


def make_expert(env: WasteSortEnv):
    """The scripted expert for the environment's task."""
    if env.task_name not in EXPERTS:
        raise KeyError(f"no scripted expert for task {env.task_name!r}; experts exist for: {sorted(EXPERTS)}")
    return EXPERTS[env.task_name](env)

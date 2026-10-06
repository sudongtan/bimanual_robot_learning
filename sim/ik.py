"""Inverse kinematics for one SO-101 arm in the scene.

Damped least squares on the arm's 5 joints (pan, lift, elbow, wrist_flex,
wrist_roll). Targets a point fixed in the gripper site frame
(`<side>_gripperframe`) and, optionally, the direction of that site's z axis,
which points out of the fingers. 3 position + 2 direction constraints = 5,
so wrist roll is determined by the solution, not chosen.

Works on its own MjData so it never disturbs the simulation.
"""

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str):
        self.m = model
        self.d = mujoco.MjData(model)
        self.side = side
        jids = [model.joint(f"{side}_{j}").id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo, self.hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.site = model.site(f"{side}_gripperframe").id
        self.body = model.site_bodyid[self.site]

    def point(self, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        """World position of `offset` (site frame) at the current IK state."""
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site] + R @ np.asarray(offset)

    def solve(self, qpos: np.ndarray, target: np.ndarray, offset=(0.0, 0.0, 0.0),
              direction: np.ndarray | None = None, dir_weight: float = 0.3,
              iters: int = 300, damping: float = 1e-3, tol: float = 1e-4) -> tuple[np.ndarray, float, float]:
        """Return (5 arm joint angles, position error m, direction error rad).

        `qpos` is the full model qpos to start from (other joints stay as given).
        """
        d, m = self.d, self.m
        d.qpos[:] = qpos
        offset = np.asarray(offset, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.site].reshape(3, 3)
            p = d.site_xpos[self.site] + R @ offset
            err_p = target - p
            rows_e, rows_J = [err_p], []
            mujoco.mj_jac(m, d, jacp, jacr, p, self.body)
            rows_J.append(jacp[:, self.dadr])
            if direction is not None:
                z = R[:, 2]
                err_r = np.cross(z, direction / np.linalg.norm(direction))
                rows_e.append(dir_weight * err_r)
                rows_J.append(dir_weight * jacr[:, self.dadr])
            e, J = np.concatenate(rows_e), np.vstack(rows_J)
            if np.linalg.norm(e) < tol:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[self.site].reshape(3, 3)
        pos_err = np.linalg.norm(target - (d.site_xpos[self.site] + R @ offset))
        dir_err = 0.0 if direction is None else float(np.arccos(np.clip(R[:, 2] @ (direction / np.linalg.norm(direction)), -1, 1)))
        return d.qpos[self.qadr].copy(), float(pos_err), dir_err

    def collides(self, qpos: np.ndarray, ignore_bodies: set[int] = frozenset()) -> list[tuple[str, str]]:
        """Contacts involving this arm at `qpos` (full model qpos), minus ignored bodies."""
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        arm = {b for b in range(m.nbody) if m.body(b).name.startswith(self.side + "_")}
        out = []
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ({b1, b2} & arm) and not ({b1, b2} & set(ignore_bodies)):
                out.append((m.body(b1).name, m.body(b2).name))
        return sorted(set(out))

"""IK reach helper shared by the scene tests and the per-task tests."""

import numpy as np

TIP = (0.012, 0, 0.003)          # fingertip, gripper-site frame
CAN_CENTRE = (-0.013, 0, 0.021)  # held can's centre, gripper-site frame
DOWN = np.array([0, 0, -1.0])


def reachable(ik, base, target, offset, direction=None, prev=None, restarts=15, max_tilt=None, ignore_bodies=()):
    """Joint angles reaching `target` without contacts, or None. `max_tilt` (deg) caps how far the
    gripper's pointing direction may be from straight down."""
    rng = np.random.default_rng(0)
    for k in range(restarts):
        q0 = base.copy()
        if k == 0 and prev is not None:
            q0[ik.qadr] = prev
        elif k:
            q0[ik.qadr] = rng.uniform(ik.lo, ik.hi)
        q, ep, er = ik.solve(q0, target, offset, direction)
        full = base.copy()
        full[ik.qadr] = q
        ok = ep < 3e-3 and er < np.radians(2) and not ik.collides(full, ignore_bodies=set(ignore_bodies))
        if ok and max_tilt is not None:
            z_axis = ik.d.site_xmat[ik.site].reshape(3, 3)[:, 2]  # ik.d holds `full` after collides()
            ok = np.degrees(np.arccos(np.clip(-z_axis[2], -1, 1))) <= max_tilt
        if ok:
            return q
    return None

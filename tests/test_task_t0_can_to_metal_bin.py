"""Checks specific to task T0, t0_can_to_metal_bin (T1 with only the can on the table). Checks shared by all tasks are in tests/test_env.py
(environment, expert) and tests/test_scene.py (scene)."""

import numpy as np
import pytest

from sim.ik import ArmIK
from sim.waste_env import CAN_PATCH, WasteSortEnv
from reach import DOWN, reachable


@pytest.fixture(scope="module")
def env():
    e = WasteSortEnv(task="t0_can_to_metal_bin")
    yield e
    e.close()


def test_can_starts_inside_its_patch(env):
    for seed in range(10):
        env.reset(seed=seed)
        x, y, _ = env.item_pos("tomato_soup_can")
        assert CAN_PATCH["x"][0] <= x <= CAN_PATCH["x"][1] and CAN_PATCH["y"][0] <= y <= CAN_PATCH["y"][1], seed


def test_can_rests_upright_where_placed(env):
    """With the arms holding home for 2 s, the can neither moves nor tips."""
    home = env.model.key_ctrl[env.home_key]
    for seed in range(5):
        env.reset(seed=seed)
        p0 = env.item_pos("tomato_soup_can")
        for _ in range(100):
            env.step(home)
        b = env.model.body("ycb_tomato_soup_can").id
        assert np.linalg.norm(env.item_pos("tomato_soup_can") - p0) < 0.005, seed
        tilt = np.degrees(np.arccos(np.clip(env.data.xmat[b].reshape(3, 3)[2, 2], -1, 1)))
        assert tilt < 5, seed


@pytest.mark.slow
@pytest.mark.parametrize("corner", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_arm_a_grasp_pose_at_can_patch_corners(env, corner):
    """Vertical gripper at grasp height around the can, at each corner of CAN_PATCH. (The approach
    from above is impossible — vertical reach ends below the can's top, debug-log #13 — so the
    expert slides in from the side at this height.)"""
    m = env.model
    xy = (CAN_PATCH["x"][corner[0]], CAN_PATCH["y"][corner[1]])
    home = m.key_qpos[env.home_key].copy()
    can = {m.body("ycb_tomato_soup_can").id}   # the keyframe leaves the can at the world origin; ignore it
    assert reachable(ArmIK(m, "left"), home, np.array([*xy, 0.37]), (-0.017, 0, 0), DOWN, ignore_bodies=can) is not None


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1004, 1009])
def test_expert_grasps_near_arm_a_base(env, seed):
    """Cans in the patch corner nearest arm A's base: sliding in from the far side let the wrist
    camera housing knock the can over (debug-log #28). The expert must grasp and deliver it."""
    from sim.experts import make_expert
    from sim.waste_env import CONTROL_HZ
    expert = make_expert(env)
    env.reset(seed=seed)
    can0 = env.item_pos("tomato_soup_can")
    plan = expert.reset()
    info = {"is_success": False}
    for k in range(int((plan.duration + 1.5) * CONTROL_HZ)):
        *_, terminated, truncated, info = env.step(expert.act(k))
        if plan.phase(k / CONTROL_HZ).startswith(expert.grasp_phase):
            assert np.linalg.norm(env.item_pos("tomato_soup_can")[:2] - can0[:2]) < 0.01, "can moved before the grasp"
        if terminated or truncated:
            break
    assert info["is_success"]

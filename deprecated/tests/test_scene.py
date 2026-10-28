"""Scene matches README section 2.2, and the cheap requirement checks from sim/check_requirements.py.

The full requirement report (minutes): uv run python -m sim.check_requirements
"""

import numpy as np
import pytest

from sim.check_requirements import ITEMS, carry_trial, drawer_trial, handover_trial
from sim.expert import task1_episode
from sim.region_map import X_RANGE, Y_RANGE, sample_start
from sim.scene import ARMS, HOME_TIP_LEFT, Sim, load_model


@pytest.fixture(scope="module")
def sim():
    return Sim(seed=0, noise=0)


def test_model_layout():
    m = load_model()
    assert m.nu == 12
    assert {m.camera(i).name for i in range(m.ncam)} == {"overview", "left/wrist_cam", "right/wrist_cam"}
    assert m.cam_fovy[m.camera("overview").id] == 45
    assert m.cam_fovy[m.camera("left/wrist_cam").id] == 75
    assert m.body_mass[m.body("can").id] == pytest.approx(0.08)
    assert m.body_mass[m.body("block").id] == pytest.approx(0.009)


def test_home_pose_matches_spec(sim):
    np.testing.assert_allclose(sim.tip("left"), HOME_TIP_LEFT, atol=2e-3)
    np.testing.assert_allclose(sim.tip("right"), HOME_TIP_LEFT * [-1, -1, 1], atol=2e-3)
    a = sim.arms["left"]
    assert np.min(np.minimum(sim.home - a.lo, a.hi - sim.home)) > 0.2  # start noise never clips


def test_arms_apart_at_home_with_start_noise():
    s = Sim(seed=0)
    for seed in range(20):
        s.reset(seed)
        assert not s.arms_touching()
        assert s.arm_clearance(distmax=0.5) > 0.10


def test_drawers_closed_and_still_at_rest(sim):
    sim.reset(0)
    sim.clear_items()
    sim.step(500)
    for d in ("metal", "plastic"):
        assert abs(sim.drawer_opening(d)) < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("drawer", ["metal", "plastic"])
def test_B_each_arm_opens_each_drawer(arm, drawer):
    r = drawer_trial(arm, drawer, seed=0)
    assert r["min_open_ok"], r
    assert r["arm_contact_steps"] == 0, r


@pytest.mark.slow
@pytest.mark.parametrize("item,xy", [("block", (-0.08, 0.18)), ("can", (0.10, -0.18))])
def test_C_item_held_while_carried(item, xy):
    arm = "left" if xy[0] < 0 else "right"
    r = carry_trial(arm, item, xy)
    assert r["result"] == "ok" and r["held"], r


@pytest.mark.slow
@pytest.mark.parametrize("giver", ARMS)
def test_A2_handover(giver):
    r = handover_trial(giver, seed=0)
    assert r["result"] == "ok" and r["taker_holds_after"], r


def test_start_region_samples_inside_patches():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = sample_start(rng)
        assert X_RANGE[0] <= abs(x) <= X_RANGE[1] and Y_RANGE[0] <= abs(y) <= Y_RANGE[1]


@pytest.mark.slow
@pytest.mark.parametrize("item", list(ITEMS))
@pytest.mark.parametrize("seed", [0, 1])
def test_task1_expert_from_start_region(item, seed):
    xy = sample_start(np.random.default_rng(seed))
    r = task1_episode(item, xy, seed=seed)
    assert r["success"], r
    assert r["max_jump"] < 0.5 and r["disturbed_cm"] < 1.0, r

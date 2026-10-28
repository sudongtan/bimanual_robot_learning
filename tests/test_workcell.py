"""Workcell scene checks for the task set in 1tasks_scene.md (S1-S4, T1, T2).

Fast tests check layout, rest behaviour and that every pose a task needs is reachable by
BOTH arms with a collision-free open gripper (commands can name either arm).
"""

import itertools

import numpy as np
import pytest

from sim import tasks as T
from sim.scene import ARMS, DRAWER_PULL, HOME_TIP_LEFT, TABLE_Z, Sim, load_model


@pytest.fixture
def sim():
    s = Sim(seed=0, noise=0)
    s.clear_items()
    return s


def corners_and_centre(area, inset=0.0):
    (x0, x1), (y0, y1) = area
    xs, ys = (x0 + inset, (x0 + x1) / 2, x1 - inset), (y0 + inset, (y0 + y1) / 2, y1 - inset)
    return list(itertools.product(xs, ys))


# --- layout ------------------------------------------------------------------


def test_model_layout():
    m = load_model()
    assert m.nu == 12
    assert {m.camera(i).name for i in range(m.ncam)} == {"overview", "left/wrist_cam", "right/wrist_cam"}
    assert m.jnt_range[m.joint("drawer").id][1] == pytest.approx(DRAWER_PULL)
    assert m.geom_size[m.geom("cube").id] == pytest.approx([T.CUBE_W / 2, T.CUBE_W / 2, T.CUBE_H / 2])


def test_home_pose(sim):
    np.testing.assert_allclose(sim.tip("left"), HOME_TIP_LEFT, atol=2e-3)
    np.testing.assert_allclose(sim.tip("right"), HOME_TIP_LEFT * [-1, -1, 1], atol=2e-3)
    a = sim.arms["left"]
    assert np.min(np.minimum(sim.home - a.lo, a.hi - sim.home)) > 0.2


def test_arms_apart_at_home_with_start_noise():
    s = Sim(seed=0)
    for seed in range(20):
        s.reset(seed)
        assert not s.arms_touching()
        assert s.arm_clearance(distmax=0.5) > 0.10


def test_nothing_touches_at_rest(sim):
    T.place_cube_on_table(sim, (0.0, -0.18))
    sim.step(250)
    for c in sim.data.contact[: sim.data.ncon]:
        pair = {sim.model.body(sim.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
        if "world" in pair:  # unused items are parked on the floor
            continue
        assert pair == {"table", "cube"}, pair


def test_drawer_stays_where_it_is_left(sim):
    sim.step(500)
    assert abs(sim.drawer_opening()) < 0.003
    sim.set_drawer(T.DRAWER_OPEN)
    sim.step(500)
    assert sim.drawer_opening() == pytest.approx(T.DRAWER_OPEN, abs=0.003)


# --- reach: every pose a task needs, for both arms -----------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_S1_S2_handle_graspable_over_full_pull(sim, arm):
    for opening in np.arange(0.0, T.DRAWER_OPEN + 1e-9, 0.02):
        assert T.plan_handle_grasp(sim, arm, opening=opening) is not None, opening


@pytest.mark.parametrize("arm", ARMS)
def test_S3_T1_cube_graspable_in_table_area(sim, arm):
    for xy in corners_and_centre(T.TABLE_AREA):
        T.place_cube_on_table(sim, xy)
        assert T.plan_cube_grasp(sim, arm) is not None, xy


@pytest.mark.parametrize("arm", ARMS)
def test_cube_targets_graspable(sim, arm):
    for name, xy in T.CUBE_TARGETS.items():
        T.place_cube_on_table(sim, xy)
        assert T.plan_cube_grasp(sim, arm) is not None, name


@pytest.mark.parametrize("arm", ARMS)
def test_S4_T2_cube_graspable_in_open_drawer(sim, arm):
    sim.set_drawer(T.DRAWER_OPEN)
    (x0, x1), (y0, y1) = T.DRAWER_AREA
    for x, y in corners_and_centre(T.DRAWER_AREA):
        T.place_cube_in_drawer(sim, (x, y - T.DRAWER_OPEN))
        assert T.plan_cube_grasp(sim, arm) is not None, (x, y)


def test_drawer_area_is_inside_the_drawer(sim):
    """The in-drawer area, at any start opening, lies inside the drawer's interior."""
    for opening in (0.0, T.DRAWER_OPEN):
        sim.set_drawer(opening)
        for x, y in corners_and_centre(T.DRAWER_AREA):
            p = np.array([x, y - opening, T.DRAWER_FLOOR_Z + T.CUBE_H / 2])
            site = sim.data.site("drawer_interior")
            local = site.xmat.reshape(3, 3).T @ (p - site.xpos)
            assert np.all(np.abs(local[:2]) <= sim.model.site("drawer_interior").size[:2] - T.CUBE_W / 2 + 1e-9), (opening, x, y)


def test_table_area_clear_of_open_drawer(sim):
    """A cube anywhere in the table area doesn't touch the drawer or handle at full opening."""
    sim.set_drawer(DRAWER_PULL)
    for xy in corners_and_centre(T.TABLE_AREA):
        T.place_cube_on_table(sim, xy)
        sim.step(1)
        assert not any(
            {sim.model.body(sim.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)} == {"cube", "drawer"}
            for c in sim.data.contact[: sim.data.ncon]
        ), xy


def test_area_sampling_stays_inside():
    rng = np.random.default_rng(0)
    (x0, x1), (y0, y1) = T.TABLE_AREA
    for _ in range(200):
        x, y = T.table_xy(rng)
        assert x0 <= x <= x1 and y0 <= y <= y1
        assert np.hypot(x - T.S3_TARGET[0], y - T.S3_TARGET[1]) >= T.START_CLEARANCE

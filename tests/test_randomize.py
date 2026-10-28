"""Per-episode randomisation (sim/randomize.py, Randomisation in 1tasks_scene.md).

Checks that every draw stays inside its stated range, that distractors sit on the table clear of
everything, that the train/eval splits don't share colours or phrasings, and that a rotated cube
is still graspable by both arms everywhere it can start.
"""

import numpy as np
import pytest

from sim import randomize as RZ
from sim import tasks as T
from sim.scene import ARMS, TABLE_Z, Sim
from test_workcell import corners_and_centre

SEEDS = range(30)


def fresh(seed, cfg=RZ.Randomization()):
    s = Sim(seed=seed)
    s.clear_items()
    return s, RZ.apply(s, seed, cfg)


def test_same_seed_same_scene():
    assert fresh(3)[1] == fresh(3)[1]
    assert fresh(3)[1] != fresh(4)[1]


def test_none_changes_nothing():
    s, info = fresh(0, RZ.NONE)
    ref = Sim(seed=0)
    assert info == {} and s.cube_yaw == 0.0 and s.distractor_xy == {}
    np.testing.assert_array_equal(s.model.geom_rgba, ref.model.geom_rgba)
    np.testing.assert_array_equal(s.model.light_dir, ref.model.light_dir)


def test_noise_settings_reach_the_sim():
    from sim.skills import new_sim

    quiet, _ = new_sim(0, RZ.NONE)
    np.testing.assert_array_equal(quiet.q("left"), quiet.home)
    assert quiet.demo_noise == 0.0
    noisy, _ = new_sim(0, RZ.Randomization())
    assert np.abs(noisy.q("left") - noisy.home).max() > 0.005
    assert noisy.demo_noise == RZ.Randomization().demo_noise


def test_start_positions_independent_of_randomisation():
    """The randomisation has its own stream: item starts and arm start noise don't change."""
    a, _ = fresh(5, RZ.NONE)
    b, _ = fresh(5)
    np.testing.assert_array_equal(a.q("left"), b.q("left"))
    assert T.table_xy(np.random.default_rng(5)) == T.table_xy(np.random.default_rng(5))


@pytest.mark.parametrize("split", ("train", "eval"))
def test_colours(split):
    palette = RZ.TRAIN_COLOURS if split == "train" else RZ.EVAL_COLOURS
    for seed in SEEDS:
        _, info = fresh(seed, RZ.Randomization(split=split))
        assert info["cube_colour"] in palette and info["bar_colour"] in palette
        assert info["cube_colour"] != info["bar_colour"]
        for d in info["distractors"]:
            assert d["colour"] in palette and d["colour"] not in (info["cube_colour"], info["bar_colour"])
    assert not set(RZ.TRAIN_COLOURS) & set(RZ.EVAL_COLOURS)


def test_cube_yaw_in_range():
    infos = [fresh(seed)[1] for seed in SEEDS]
    yaws = [i["cube_yaw"] for i in infos]
    assert all(abs(y) <= 45 for y in yaws) and all(abs(i["cube_yaw_drawer"]) <= 20 for i in infos)
    assert max(yaws) - min(yaws) > 45  # actually spread out


def test_distractors_on_table_and_clear_of_everything():
    counts = set()
    for seed in SEEDS:
        s, info = fresh(seed)
        counts.add(len(info["distractors"]))
        s.step(250)
        for body, xy in s.distractor_xy.items():
            assert np.linalg.norm(s.body_pos(body)[:2] - xy) < 0.001, (seed, body)  # settled, not pushed
            assert s.body_pos(body)[2] > TABLE_Z
        for c in s.data.contact[: s.data.ncon]:
            pair = {s.model.body(s.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
            if "world" in pair:  # parked items on the floor
                continue
            assert len(pair) == 2 and "table" in pair, (seed, pair)  # only resting on the table
    assert counts == {0, 1, 2}


def test_distractor_zones_clear_of_task_areas():
    pad = 0.025 + RZ.DISTRACTOR_MOVED  # largest distractor radius + margin
    task_areas = [T.TABLE_AREA, T.BAR_AREA] + [((x, x), (y, y)) for x, y in T.CUBE_TARGETS.values()]
    for (zx0, zx1), (zy0, zy1) in RZ.DISTRACTOR_ZONES:
        for (ax0, ax1), (ay0, ay1) in task_areas:
            gap_x = max(zx0 - ax1, ax0 - zx1)
            gap_y = max(zy0 - ay1, ay0 - zy1)
            assert max(gap_x, gap_y) > T.CUBE_W / 2 + pad


@pytest.mark.parametrize("task", sorted(RZ.COMMANDS))
def test_commands(task):
    names = dict(arm="A", target="left", giver="A", taker="B", opener="A", picker="B")
    canonical = RZ.command(task, 0, RZ.NONE, **names)
    assert canonical == RZ.COMMANDS[task][0].format(**names)
    train = {RZ.command(task, s, RZ.Randomization(), **names) for s in SEEDS}
    held_out = RZ.command(task, 0, RZ.Randomization(split="eval"), **names)
    assert canonical in train and held_out not in train and len(train) == len(RZ.COMMANDS[task]) - 1


@pytest.mark.parametrize("u", (-1, -0.5, 0.5, 1))
@pytest.mark.parametrize("arm", ARMS)
def test_rotated_cube_graspable(arm, u):
    """At the ends of both yaw ranges (±45° on the table, ±20° in the drawer)."""
    cfg = RZ.Randomization()
    s = Sim(seed=0, noise=0)
    s.clear_items()
    s.cube_yaw, s.cube_yaw_drawer = np.deg2rad(u * cfg.cube_yaw), np.deg2rad(u * cfg.cube_yaw_drawer)
    for xy in corners_and_centre(T.TABLE_AREA):
        T.place_cube_on_table(s, xy)
        assert T.plan_cube_grasp(s, arm) is not None, ("table", xy)
    s.set_drawer(T.DRAWER_OPEN)
    for x, y in corners_and_centre(T.DRAWER_AREA):
        T.place_cube_in_drawer(s, (x, y - T.DRAWER_OPEN))
        assert T.plan_cube_grasp(s, arm) is not None, ("drawer", x, y)

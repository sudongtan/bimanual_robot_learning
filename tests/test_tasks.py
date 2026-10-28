"""End-to-end task runs with the scripted skills (sim/skills.py), both arms, random starts.

Each case plays a full episode with demonstration noise and checks the task's success
condition plus no arm-to-arm contact and no posture jumps. Larger validation runs:
`python -m sim.validate`.
"""

import pytest

from sim import skills as K

SEEDS = (0, 1)
MAX_JUMP = 0.15  # rad between IK waypoints; larger means the arm switched posture mid-move (largest seen 0.11)


def check(r):
    assert r["success"], r
    assert r["arm_contacts"] == 0, r
    assert r["pushed"] < K.PUSHED, r
    assert r["max_jump"] < MAX_JUMP, r


@pytest.mark.slow
@pytest.mark.parametrize("seed, in_drawer", ((0, False), (2, True)))
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S1_open_drawer(arm, seed, in_drawer):
    """Both S1 starts: cube on the table, and the T2 start (cube in the closed drawer)."""
    r = K.run_S1(arm, seed=seed)
    assert r["cube_in_drawer"] == in_drawer, r
    check(r)


@pytest.mark.slow
@pytest.mark.parametrize("seed, in_drawer", ((0, False), (2, True)))
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S2_close_drawer(arm, seed, in_drawer):
    """Both S2 starts: cube on the table, and the S4 start (cube in the open drawer)."""
    r = K.run_S2(arm, seed=seed)
    assert r["cube_in_drawer"] == in_drawer, r
    check(r)


@pytest.mark.slow
@pytest.mark.parametrize("seed, drawer_open", ((0, False), (2, True)))
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S3_move_cube(arm, seed, drawer_open):
    """Both S3 starts: drawer closed, and drawer partly open (S2's cube-on-table start)."""
    r = K.run_S3(arm, seed=seed)
    assert r["drawer_open"] == drawer_open, r
    check(r)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("target", ("left", "middle", "right"))
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S4_cube_out_of_drawer(arm, target, seed):
    check(K.run_S4(arm, target, seed=seed))


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("target", ("left", "middle", "right"))
@pytest.mark.parametrize("opener", ("A", "B"))
def test_T2_open_then_take_out(opener, target, seed):
    check(K.run_T2(opener, target, seed=seed))


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("giver", ("A", "B"))
def test_T1_bar_handoff(giver, seed):
    check(K.run_T1(giver, seed=seed))

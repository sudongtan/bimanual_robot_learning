"""Starts and success checks that apply to any episode (sim/episode.py), whoever acts.

The expert passing every check is covered in test_tasks.py; here, the checks must fail when the
task isn't done as commanded: nothing done, the wrong arm, the cube dropped.
"""

import numpy as np
import pytest

from sim import episode as E
from sim import skills as K
from sim import tasks as T
from sim.randomize import VARIANTS, Randomization


def scene(sim):
    return np.concatenate([sim.body_pos("cube"), [sim.drawer_opening()], sim.q("left"), sim.q("right")])


@pytest.mark.parametrize("seed", (0, 1, 2, 3))
def test_shared_scenes_same_start(seed):
    """Variants that share a start scene get the same start for the same seed (1tasks_scene.md)."""
    def start(variant):
        return scene(E.reset(variant, seed)[0])

    np.testing.assert_array_equal(start(("S1", "A", None)), start(("T1", "B", "right")))
    np.testing.assert_array_equal(start(("S3", "A", None)), start(("T2", "B", None)))


@pytest.mark.parametrize("variant", VARIANTS)
def test_doing_nothing_fails(variant):
    """Both arms holding their start pose for 2 s: no variant counts as done."""
    sim, start = E.reset(variant, 0)
    sim.step(1000)
    ok, checks = E.success(sim, start)
    assert not ok, checks
    assert checks["no_arm_contact"] and checks["distractors_still"]


@pytest.mark.slow
def test_wrong_arm_fails():
    """S1 told arm A, done by arm B: the drawer is open, but the check fails on who did it."""
    sim, start = E.reset(("S1", "A", None), 0)
    K.move_drawer(sim, "right", T.DRAWER_OPEN)
    ok, checks = E.success(sim, start)
    assert checks["drawer_open"] and not ok
    assert not checks["named_arm_acted"] and not checks["other_arm_stayed_home"]


@pytest.mark.slow
def test_dropped_cube_is_caught():
    """The cube let go at carry height falls far with no arm touching it: not set down."""
    sim, start = E.reset(("S3", "A", None), 0, Randomization(distractors=0))
    assert K.pick_cube(sim, "left")
    sim.grip("left", T.CUBE_OPEN, 0.5)
    sim.step(500)
    assert sim.monitor.drop > 0.04
    assert not E.success(sim, start)[1]["set_down"]

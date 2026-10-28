"""End-to-end task runs with the scripted skills (sim/skills.py), both arms, random starts.

Each case plays a full episode with demonstration noise and checks the task's success
condition plus no arm-to-arm contact, no posture jumps and a still hold at the end. Larger validation runs:
`python -m sim.validate`.
"""

import mujoco
import numpy as np
import pytest

from sim import skills as K
from sim.scene import CONTROL_EVERY, STATE_SPEC

SEEDS = (0, 1)
MAX_JUMP = 0.15  # rad between IK waypoints; larger means the arm switched posture mid-move (largest seen 0.11)


def check(r):
    assert r["success"], r
    assert r["arm_contacts"] == 0, r
    assert r["pushed"] < K.PUSHED, r
    assert r["max_jump"] < MAX_JUMP, r
    assert r["hold_motion"] < K.HOLD_STILL, r
    assert r["hold_moved"] < K.PUSHED, r


@pytest.mark.slow
def test_hold_is_recorded():
    """The recorder sees the HOLD_S hold at the end of the episode, with both arms still at home
    (so "stay" after the task is in the training data: the stop tests in 1tasks_scene.md)."""
    qs = []
    r = K.run_S1("A", seed=0, rec=lambda s: qs.append(np.concatenate([s.q(a) for a in s.arms])))
    check(r)
    n = int(K.HOLD_S / 0.002)
    tail = np.array(qs[-n:])
    assert len(qs) > n
    assert np.abs(tail - tail[0]).max() < K.HOLD_STILL


class Frames:
    """Records what the recorder will: at each control tick, the state before it, the held command and the subtask."""

    def __init__(self):
        self.states, self.actions, self.nominal, self.subtask, self.held = [], [], [], [], True
        self.sim = None

    def __call__(self, s):
        self.sim = s
        if s.ticked:
            self.states.append(s.tick_state.copy())
            self.actions.append(s.applied_ctrl.copy())
            self.nominal.append(s.nominal_ctrl.copy())
            self.subtask.append(s.subtask)
        elif self.actions:
            self.held &= bool(np.array_equal(s.applied_ctrl, self.actions[-1]))


@pytest.mark.slow
@pytest.mark.parametrize("run", (lambda rec: K.run_S1("A", seed=0, rec=rec),
                                 lambda rec: K.run_T1("B", "left", seed=1, rec=rec)), ids=("S1", "T1"))
def test_replay_is_exact(run):
    """Each command is held for CONTROL_EVERY steps (50 Hz), and replaying the recorded commands from
    the recorded start reproduces every recorded state (2data_collection.md, first-batch check 1)."""
    f = Frames()
    check(run(f))
    assert f.held
    assert any(np.abs(a - n).max() > 0 for a, n in zip(f.actions, f.nominal))  # demonstration noise is in `action`
    model, spec = f.sim.model, STATE_SPEC
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, f.states[0], spec)
    state = np.empty_like(f.states[0])
    for k, action in enumerate(f.actions[:-1]):
        data.ctrl[:] = action
        mujoco.mj_step(model, data, CONTROL_EVERY)
        mujoco.mj_getState(model, data, state, spec)
        assert np.array_equal(state, f.states[k + 1]), k


@pytest.mark.slow
@pytest.mark.parametrize("run", (lambda rec: K.run_T1("A", "middle", seed=0, rec=rec),
                                 lambda rec: K.run_T2("B", seed=0, rec=rec)), ids=("T1", "T2"))
def test_composite_subtask_boundary(run):
    """A composite's frames are 0 (first arm's step) then 1 (second arm's step), switching once."""
    f = Frames()
    check(run(f))
    sub = np.array(f.subtask)
    assert sub[0] == 0 and sub[-1] == 1 and np.count_nonzero(np.diff(sub)) == 1


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S1_open_drawer(arm, seed):
    """S1 starts like T1: cube inside the closed drawer."""
    check(K.run_S1(arm, seed=seed))


@pytest.mark.slow
@pytest.mark.parametrize("seed, in_drawer", ((0, False), (2, True)))
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S2_close_drawer(arm, seed, in_drawer):
    """Both S2 starts: the S3/T2 start (partly open, cube on the table), and the S4 start (fully open, cube inside)."""
    r = K.run_S2(arm, seed=seed)
    assert r["cube_in_drawer"] == in_drawer, r
    check(r)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("arm", ("A", "B"))
def test_S3_cube_into_drawer(arm, seed):
    check(K.run_S3(arm, seed=seed))


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
def test_T1_open_then_take_out(opener, target, seed):
    check(K.run_T1(opener, target, seed=seed))


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("placer", ("A", "B"))
def test_T2_put_in_then_close(placer, seed):
    check(K.run_T2(placer, seed=seed))

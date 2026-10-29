"""The policy rollout loop (sim/evaluate.py), driven by the expert's own commands instead of a policy."""

import dataclasses

import numpy as np
import pytest
import torch

from sim import evaluate as V
from sim import randomize as RZ
from sim import record as R
from sim import skills as K
from sim.record import Frames


class Replay:
    """A stand-in policy: the recorded commands in order, then the last one held."""

    def __init__(self, actions):
        self.actions, self.k = actions, 0

    def reset(self):
        self.k = 0

    def select_action(self, obs):
        assert obs["observation.images.overview"].shape == (1, 3, R.HEIGHT, R.WIDTH)
        a = self.actions[min(self.k, len(self.actions) - 1)]
        self.k += 1
        return torch.from_numpy(a)[None]


@pytest.mark.slow
def test_expert_commands_succeed_in_rollout():
    """Replaying the expert's commands (no demonstration noise) through the rollout loop, from the
    same eval seed, reproduces its success and its stop: the loop resets, holds each command for
    CONTROL_EVERY steps and checks success exactly as the recording does."""
    seed = RZ.EVAL_SEEDS[0]
    rand = dataclasses.replace(RZ.Randomization(), demo_noise=0.0)
    f = Frames()
    assert K.run_S1("A", seed=seed, rec=f, rand=rand)["success"]
    identity = lambda x: x
    r, frames = V.rollout(Replay(np.array(f.actions)), identity, identity, ("S1", "A", None), seed, rand)
    assert r["success"], r
    assert r["stopped"], r
    assert not frames

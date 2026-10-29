"""The policy rollout loop (sim/evaluate.py), driven by the expert's own commands instead of a policy."""

import dataclasses

import numpy as np
import pytest
import torch

from sim import evaluate as V
from sim import randomize as RZ
from sim import record as R
from sim import skills as K
from sim.dataset_view import ENV
from sim.record import Frames


class Replay:
    """A stand-in policy: the recorded commands in order, then the last one held."""

    def __init__(self, actions, vector=None):
        self.actions, self.k = actions, 0
        self.variant_input, self.vector = vector is not None, vector

    def reset(self):
        self.k = 0

    def select_action(self, obs):
        assert obs["observation.images.overview"].shape == (1, 3, R.HEIGHT, R.WIDTH)
        if self.variant_input:
            np.testing.assert_array_equal(obs[ENV][0].numpy(), self.vector)
        a = self.actions[min(self.k, len(self.actions) - 1)]
        self.k += 1
        return torch.from_numpy(a)[None]


@pytest.mark.slow
@pytest.mark.parametrize("variant", (("S1", "A", None), ("T1", "B", "left")), ids=("S1", "T1"))
def test_expert_commands_succeed_in_rollout(variant):
    """Replaying the expert's commands (no demonstration noise) through the rollout loop, from the
    same eval seed, reproduces its success: the loop resets, holds each command for CONTROL_EVERY
    steps and checks success exactly as the recording does. S1 also stops (stop rate), T1's second
    arm goes on after the drawer is open (continue rate); the policy gets the variant vector."""
    seed = RZ.EVAL_SEEDS[0]
    rand = dataclasses.replace(RZ.Randomization(), demo_noise=0.0)
    f = Frames()
    task, arm, target = variant
    run = K.run_S1(arm, seed=seed, rec=f, rand=rand) if task == "S1" else K.run_T1(arm, target, seed=seed, rec=f, rand=rand)
    assert run["success"]
    identity = lambda x: x
    r, frames = V.rollout(Replay(np.array(f.actions), RZ.variant_vector(*variant)), identity, identity, variant, seed, rand)
    assert r["success"], r
    assert r["stopped"] if task == "S1" else r["continued"], r
    assert not frames

"""The residual-RL episode (sim/residual.py), driven by the expert's own commands as the base policy's chunks."""

import dataclasses

import numpy as np
import pytest
import torch

from sim import evaluate as V
from sim import randomize as RZ
from sim import record as R
from sim import residual as RS
from sim import skills as K
from sim.dataset_view import ENV
from sim.record import Frames


class ChunkReplay:
    """A stand-in base policy: each plan returns the next 100 recorded commands from the current frame (the last one
    held past the end), as ACT's predict_action_chunk returns a chunk."""

    image_size = (R.HEIGHT, R.WIDTH)

    def __init__(self, actions, vector, plan_every):
        self.actions, self.vector, self.plan_every, self.k = actions, vector, plan_every, 0
        self.variant_input = True

    def predict_action_chunk(self, obs):
        assert obs["observation.images.overview"].shape == (1, 3, R.HEIGHT, R.WIDTH)
        np.testing.assert_array_equal(obs[ENV][0].numpy(), self.vector)
        i = np.minimum(np.arange(self.k, self.k + 100), len(self.actions) - 1)
        self.k += self.plan_every
        return torch.from_numpy(self.actions[i])[None]


@pytest.mark.slow
def test_base_alone_and_zero_residual_reproduce_the_expert():
    """With the expert's commands as the base, an S1 episode succeeds and stops (reward 1), scored as the
    evaluation scores it; a residual that returns zeros is called every frame with features of the documented
    size and changes nothing."""
    variant, seed = ("S1", "A", None), RZ.EVAL_SEEDS[0]
    rand = dataclasses.replace(RZ.Randomization(), demo_noise=0.0)
    f = Frames()
    assert K.run_S1("A", seed=seed, rec=f, rand=rand)["success"]
    actions, vector = np.array(f.actions), RZ.variant_vector(*variant)
    identity = lambda x: x

    timing = {}
    base = RS.episode(ChunkReplay(actions, vector, RS.PLAN_EVERY), identity, identity, variant, seed, rand, timing=timing)
    assert base["success"] and base["stopped"] and base["reward"] == 1.0, base
    assert set(timing) == {"render", "base", "physics", "residual"}

    calls = []

    def zero(x):
        assert x.shape == (RS.feature_size(),)
        calls.append(1)
        return np.zeros(12)

    r = RS.episode(ChunkReplay(actions, vector, RS.PLAN_EVERY), identity, identity, variant, seed, rand, residual=zero)
    assert len(calls) == V.HORIZON["S1"]
    assert r["reward"] == base["reward"] and r["cube_slide_mm"] == base["cube_slide_mm"]

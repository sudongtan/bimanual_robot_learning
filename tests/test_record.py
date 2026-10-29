"""The recorder (sim/record.py): one short recording, without images, written and read back."""

import json

import numpy as np
import pytest

from sim import randomize as RZ
from sim import record as R


@pytest.mark.slow
def test_record_and_read_back(tmp_path):
    """Two S2 episodes (one per start kind), two T2 and two S4 (whose result has `picked`, from numpy): the dataset holds only successful episodes,
    in order, with the right variant, subtask labels and command; the log and states files match."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"
    variants = [("S2", "A", None), ("T2", "B", None), ("S4", "A", "left")]
    R.record(root, variants, 2, RZ.Randomization(), images=False, workers=2)
    log = [json.loads(line) for line in open(root / "sim" / "episodes.jsonl")]
    good = [e for e in log if e["success"]]
    assert [e["episode_index"] for e in good] == list(range(len(good)))
    assert sorted(e["cube_in_drawer"] for e in good if e["variant"][0] == "S2") == [False, True]
    assert all(e["seed"] in RZ.train_seeds(*e["variant"]) for e in log)

    ds = LeRobotDataset(R.REPO_ID, root=root)
    assert ds.num_episodes == len(good) == 6
    for e in good:
        ep = ds.meta.episodes[e["episode_index"]]
        assert ep["length"] == e["frames"]
        first, last = ds[ep["dataset_from_index"]], ds[ep["dataset_to_index"] - 1]
        assert first["task"] == e["command"]
        assert int(first["variant_index"]) == RZ.VARIANTS.index(tuple(e["variant"]))
        states = np.load(root / "sim" / e["states"])
        np.testing.assert_allclose(first["action"].numpy(), states["action"][0], atol=1e-6)
        if e["variant"][0] == "T2":
            assert int(first["subtask_index"]) == 0 and int(last["subtask_index"]) == 1

    # the training view (sim/dataset_view.py): clean labels as `action`, the variant vector as environment state
    from sim.dataset_view import ENV, view

    view(root, tmp_path / "view", nominal=True, variant_input=True)
    v = LeRobotDataset(R.REPO_ID, root=tmp_path / "view")
    assert v.num_episodes == ds.num_episodes and ENV in v.meta.stats
    for e in good:
        i = v.meta.episodes[e["episode_index"]]["dataset_from_index"]
        np.testing.assert_allclose(v[i]["action"].numpy(), ds[i]["action_nominal"].numpy())
        np.testing.assert_array_equal(v[i][ENV].numpy(), RZ.variant_vector(*e["variant"]))

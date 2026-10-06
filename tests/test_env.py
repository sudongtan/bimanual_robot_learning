"""Environment and scripted-expert tests for the can_to_metal_bin task."""

import numpy as np
import pytest

from sim.experts import EXPERTS, CanToMetalBinExpert, make_expert
from sim.waste_env import CAN_PATCH, CONTROL_HZ, TABLE_TOP, TASKS, WasteSortEnv


@pytest.fixture(scope="module")
def env():
    e = WasteSortEnv(task="can_to_metal_bin")
    yield e
    e.close()


def test_spaces_and_rate(env):
    assert env.action_space.shape == (12,)
    assert env.observation_space["state"].shape == (12,)
    assert env.n_substeps * env.model.opt.timestep == pytest.approx(1 / CONTROL_HZ)


def test_reset_is_deterministic_per_seed(env):
    env.reset(seed=3)
    a = env.data.qpos.copy()
    env.reset(seed=3)
    np.testing.assert_array_equal(a, env.data.qpos)
    env.reset(seed=4)
    assert not np.allclose(a, env.data.qpos)


def test_reset_places_only_the_task_items(env):
    for seed in range(10):
        env.reset(seed=seed)
        on_table = [env.model.body(b).name for b in range(env.model.nbody)
                    if env.model.body(b).name.startswith("ycb_") and env.data.xpos[b][2] > TABLE_TOP]
        assert on_table == ["ycb_tomato_soup_can"], seed
        x, y, _ = env.item_pos("tomato_soup_can")
        assert CAN_PATCH["x"][0] <= x <= CAN_PATCH["x"][1] and CAN_PATCH["y"][0] <= y <= CAN_PATCH["y"][1]
        assert not env.item_touches_arm("tomato_soup_can")


def test_holding_home_keeps_still_and_not_successful(env):
    obs, info = env.reset(seed=0)
    assert info["instruction"] == TASKS["can_to_metal_bin"].instruction and not info["success"]
    home = env.model.key_ctrl[env.home_key]
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(home)
    assert np.abs(obs["state"] - home).max() < 1e-3
    assert reward == 0 and not terminated and not truncated


def test_time_limit_truncates(env):
    env.reset(seed=0)
    home = env.model.key_ctrl[env.home_key]
    truncated = False
    for _ in range(env.max_steps):
        *_, truncated, _ = env.step(home)
    assert truncated


def test_camera_observations():
    e = WasteSortEnv(cameras=("overview", "left_wrist", "right_wrist"), image_size=(64, 64))
    obs, _ = e.reset(seed=0)
    assert set(obs["images"]) == {"overview", "left_wrist", "right_wrist"}
    assert all(im.shape == (64, 64, 3) and im.dtype == np.uint8 for im in obs["images"].values())
    e.close()


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 25, 47])   # 25 once knocked the can over on the approach
def test_expert_succeeds(env, seed):
    expert = CanToMetalBinExpert(env)
    env.reset(seed=seed)
    can0 = env.item_pos("tomato_soup_can")
    plan = expert.reset()
    info = {"success": False}
    for k in range(min(env.max_steps, int((plan.duration + 1.5) * CONTROL_HZ))):
        *_, terminated, truncated, info = env.step(expert.act(k))
        if plan.phase(k / CONTROL_HZ).startswith("left: close"):
            assert np.linalg.norm(env.item_pos("tomato_soup_can")[:2] - can0[:2]) < 0.01, "can moved before the grasp"
        if terminated or truncated:
            break
    assert info["success"]


@pytest.mark.slow
def test_saved_states_replay_exactly(env):
    """Rendering from saved states (scene-decisions D20) is only valid if a saved qpos reproduces
    the scene: replaying the recorded states must give the same can pose and the same success."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("record_states", "scripts/record_states.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    ep = rec.record(env, CanToMetalBinExpert(env), seed=0)
    assert ep["success"] and ep["qpos"].shape == (len(ep["action"]), env.model.nq)
    final_can = env.item_pos("tomato_soup_can")
    # replay: set each saved state on a fresh model and check the state reproduces the scene
    other = WasteSortEnv(task="can_to_metal_bin")
    other._use_items(tuple(str(i) for i in ep["items"]))
    import mujoco
    for t in (0, len(ep["qpos"]) // 2, len(ep["qpos"]) - 1):
        other.data.qpos[:] = ep["qpos"][t]
        mujoco.mj_forward(other.model, other.data)
        np.testing.assert_allclose(other.data.qpos[other.arm_qadr], ep["state"][t], atol=1e-6)
    assert other.task.success(other)                    # the last saved state is a success state
    assert np.linalg.norm(other.item_pos("tomato_soup_can") - final_can) < 0.02
    other.close()


def test_expert_registry(env):
    """Task-agnostic tools get the expert from the registry; an unknown task fails with a clear message."""
    assert isinstance(make_expert(env), EXPERTS[env.task_name])
    for cls in EXPERTS.values():
        assert isinstance(cls.grasped_item, str) and isinstance(cls.grasp_phase, str)

    class Fake:
        task_name = "no_such_task"
    with pytest.raises(KeyError, match="no scripted expert"):
        make_expert(Fake())

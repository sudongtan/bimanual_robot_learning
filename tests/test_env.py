"""Environment and scripted-expert tests that hold for every task.

Each test runs once per task in sim.waste_env.TASKS (expert tests: once per task
in sim.experts.EXPERTS). Checks that only make sense for one task live in
tests/test_task_<task>.py.
"""

import importlib.util

import mujoco
import numpy as np
import pytest

from sim.experts import EXPERTS, make_expert
from sim.waste_env import CONTROL_HZ, ITEM_PREFIX, TABLE_TOP, TASKS, WasteSortEnv


@pytest.fixture(scope="module", params=sorted(TASKS))
def env(request):
    e = WasteSortEnv(task=request.param)
    yield e
    e.close()


@pytest.fixture(scope="module", params=sorted(EXPERTS))
def expert_env(request):
    e = WasteSortEnv(task=request.param)
    yield e
    e.close()


def run_expert(env, seed):
    """Run the task's expert for its plan plus 1.5 s, stopping at success; yields (k, plan) each step."""
    expert = make_expert(env)
    env.reset(seed=seed)
    plan = expert.reset()
    info = {"is_success": False}
    for k in range(min(env.max_steps, int((plan.duration + 1.5) * CONTROL_HZ))):
        *_, terminated, truncated, info = env.step(expert.act(k))
        yield k, plan, expert, info
        if terminated or truncated:
            break


def test_spaces_and_rate(env):
    assert env.action_space.shape == (12,)
    assert env.observation_space["agent_pos"].shape == (12,)
    assert env.n_substeps * env.model.opt.timestep == pytest.approx(1 / CONTROL_HZ)


def test_reset_is_deterministic_per_seed(env):
    env.reset(seed=3)
    a = env.data.qpos.copy()
    env.reset(seed=3)
    np.testing.assert_array_equal(a, env.data.qpos)
    env.reset(seed=4)
    assert not np.allclose(a, env.data.qpos)


def test_reset_places_only_the_task_items(env):
    expected = sorted(ITEM_PREFIX + i for i in env.task.items)
    for seed in range(10):
        env.reset(seed=seed)
        on_table = sorted(env.model.body(b).name for b in range(env.model.nbody)
                          if env.model.body(b).name.startswith(ITEM_PREFIX) and env.data.xpos[b][2] > TABLE_TOP)
        assert on_table == expected, seed
        assert not any(env.item_touches_arm(i) for i in env.task.items), seed


def test_holding_home_keeps_still_and_not_successful(env):
    obs, info = env.reset(seed=0)
    assert info["instruction"] == env.task.instruction and not info["is_success"]
    home = env.model.key_ctrl[env.home_key]
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(home)
    assert np.abs(obs["agent_pos"] - home).max() < 1e-3
    assert reward == 0 and not terminated and not truncated


def test_time_limit_truncates(env):
    env.reset(seed=0)
    home = env.model.key_ctrl[env.home_key]
    truncated = False
    for _ in range(env.max_steps):
        *_, truncated, _ = env.step(home)
    assert truncated


def test_camera_observations(env):
    e = WasteSortEnv(task=env.task_name, cameras=("overview", "left_wrist", "right_wrist"), image_size=(64, 64))
    obs, _ = e.reset(seed=0)
    assert set(obs["pixels"]) == {"overview", "left_wrist", "right_wrist"}
    assert all(im.shape == (64, 64, 3) and im.dtype == np.uint8 for im in obs["pixels"].values())
    e.close()


def test_expert_registry():
    """Task-agnostic tools get the expert from the registry; an unknown task fails with a clear message."""
    for task, cls in EXPERTS.items():
        assert task in TASKS
        assert isinstance(cls.grasped_item, str) and isinstance(cls.grasp_phase, str)
        e = WasteSortEnv(task=task)
        assert isinstance(make_expert(e), cls)
        e.close()

    class Fake:
        task_name = "no_such_task"
    with pytest.raises(KeyError, match="no scripted expert"):
        make_expert(Fake())


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 25, 47])   # T0: seed 25 once knocked the can over on the approach
def test_expert_succeeds(expert_env, seed):
    env = expert_env
    start = None
    for k, plan, expert, info in run_expert(env, seed):
        if start is None:
            start = env.item_pos(expert.grasped_item)
        if plan.phase(k / CONTROL_HZ).startswith(expert.grasp_phase):
            moved = np.linalg.norm(env.item_pos(expert.grasped_item)[:2] - start[:2])
            assert moved < 0.01, f"{expert.grasped_item} moved before the grasp"
    assert info["is_success"]


@pytest.mark.slow
def test_success_still_holds_after_settling(expert_env):
    """Success is checked at one step and fires while the released item is still falling
    (debug-log #25). Run the expert to the end of its plan plus 3 s: the task's success
    condition must still hold once everything has come to rest."""
    env = expert_env
    expert = make_expert(env)
    env.reset(seed=0)
    plan = expert.reset()
    fired = False
    for k in range(int((plan.duration + 3.0) * CONTROL_HZ)):
        *_, info = env.step(expert.act(k))
        fired |= info["is_success"]
    assert fired and env.task.success(env)
    assert np.abs(env.data.qvel).max() < 0.05          # at rest, not just passing through


@pytest.mark.slow
def test_saved_states_replay_exactly(expert_env):
    """Rendering from saved states (design-decisions D20) is only valid if a saved qpos reproduces
    the scene: replaying the recorded states must give the same item poses and the same success."""
    env = expert_env
    spec = importlib.util.spec_from_file_location("record_states", "scripts/record_states.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    ep = rec.record(env, make_expert(env), seed=0)
    assert ep["success"] and ep["qpos"].shape == (len(ep["action"]), env.model.nq)
    final = {i: env.item_pos(i) for i in env.task.items}
    # replay: set saved states on a fresh environment and check they reproduce the scene
    other = WasteSortEnv(task=env.task_name)
    other._use_items(tuple(str(i) for i in ep["items"]))
    for t in (0, len(ep["qpos"]) // 2, len(ep["qpos"]) - 1):
        other.data.qpos[:] = ep["qpos"][t]
        mujoco.mj_forward(other.model, other.data)
        np.testing.assert_allclose(other.data.qpos[other.arm_qadr], ep["state"][t], atol=1e-6)
    assert other.task.success(other)                    # the last saved state is a success state
    for i, p in final.items():
        assert np.linalg.norm(other.item_pos(i) - p) < 0.02, i
    other.close()


def test_lerobot_creates_the_env():
    """LeRobot's evaluator builds the env from the `waste_sort` config (sim/lerobot_plugin) and reads
    `agent_pos`, `pixels/<camera>`, `info["is_success"]` and `task_description`."""
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import preprocess_observation
    from sim.lerobot_plugin import WasteSortEnvConfig

    for task in TASKS:
        cfg = WasteSortEnvConfig(task=task, cameras=["overview"], observation_height=32, observation_width=32)
        vec = make_env(cfg, n_envs=1)["waste_sort"][0]
        obs, info = vec.reset(seed=[0])
        batch = preprocess_observation(obs)
        assert batch["observation.state"].shape == (1, 12)
        assert batch["observation.images.overview"].shape == (1, 3, 32, 32)
        assert list(vec.call("task_description")) == [TASKS[task].instruction]
        assert vec.call("_max_episode_steps")[0] == int(TASKS[task].max_seconds * CONTROL_HZ)
        *_, info = vec.step(np.zeros((1, 12), np.float32))
        assert "is_success" in info
        vec.close()

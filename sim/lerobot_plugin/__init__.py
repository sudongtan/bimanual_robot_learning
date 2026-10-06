"""Registers the waste-sorting environment with LeRobot, so LeRobot's evaluator
(`lerobot-eval`, and the evaluation inside `lerobot-train`) can run it.

LeRobot finds simulation environments through an `EnvConfig` subclass registered
under a name (here `waste_sort`), which creates the gym environment by id. Loaded
with `--env.discover_packages_path=sim.lerobot_plugin`; LeRobot imports this
package, which registers both (LeRobot's `load_plugin` in lerobot/configs/parser.py).

Gym ids: `gym_waste_sort/<task>`, one per task in sim.waste_env.TASKS.
"""

from dataclasses import dataclass, field

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import freeze_after_episode_end
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from sim.waste_env import CONTROL_HZ, TASKS

GYM_NAMESPACE = "gym_waste_sort"
N_JOINTS = 12

for _name, _task in TASKS.items():
    _id = f"{GYM_NAMESPACE}/{_name}"
    if _id not in gym.registry:
        gym.register(_id, entry_point="sim.waste_env:WasteSortEnv", kwargs={"task": _name},
                     max_episode_steps=int(_task.max_seconds * CONTROL_HZ))


@EnvConfig.register_subclass("waste_sort")
@dataclass
class WasteSortEnvConfig(EnvConfig):
    task: str | None = "t0_can_to_metal_bin"
    fps: int = CONTROL_HZ
    cameras: list[str] = field(default_factory=lambda: ["overview", "left_wrist", "right_wrist"])
    observation_height: int = 224
    observation_width: int = 224
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Same names and shapes as the recorded dataset (scripts/render_dataset.py).
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(N_JOINTS,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(N_JOINTS,)),
            **{f"pixels/{c}": PolicyFeature(type=FeatureType.VISUAL,
                                            shape=(self.observation_height, self.observation_width, 3))
               for c in self.cameras},
        }
        self.features_map = {ACTION: ACTION, "agent_pos": OBS_STATE,
                             **{f"pixels/{c}": f"{OBS_IMAGES}.{c}" for c in self.cameras}}

    @property
    def package_name(self) -> str:
        return GYM_NAMESPACE

    @property
    def gym_kwargs(self) -> dict:
        return {"cameras": tuple(self.cameras), "image_size": (self.observation_height, self.observation_width)}

    def create_envs(self, n_envs: int, use_async_envs: bool = False) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """As EnvConfig.create_envs, plus LeRobot's FreezeAfterEpisodeEnd wrapper: with the default
        SAME_STEP autoreset a finished episode is reset at its last step, so the final state (item in
        the bin) is never rendered; the wrapper replays the final step instead, and a finished
        sub-env stops simulating while the rest of the batch runs."""
        def make_one():
            return gym.make(self.gym_id, disable_env_checker=self.disable_env_checker, **self.gym_kwargs)

        fns = [freeze_after_episode_end(make_one) for _ in range(n_envs)]
        if use_async_envs and n_envs > 1:
            vec = gym.vector.AsyncVectorEnv(fns, context="forkserver", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        else:
            vec = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        return {self.type: {0: vec}}

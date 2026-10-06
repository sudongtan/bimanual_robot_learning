"""Gymnasium environment for the table waste-sorting scene.

One environment, many tasks: the scene, action, observation and control rate
are shared; a task only sets its instruction, which items it uses, how
`reset` places them, its success check and its time limit (see TASKS).

The scene file holds no items. On `reset` the environment attaches the
task's items (YCB models in sim/assets/objects/ycb/) with MjSpec and compiles
the result; one compiled model is kept per item set (design-decisions D17).

    env = WasteSortEnv(task="t0_can_to_metal_bin")
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(action)

Action: 12 absolute joint position targets in radians, in actuator order
(left_ then right_: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper) — the same quantity the SO-101 position servos take.
Applied for one control step = 1/CONTROL_HZ s.

Observation: {"agent_pos": the 12 arm joint positions} and, if `cameras` is
given, {"pixels": {camera: HxWx3 uint8}}. Info: "is_success", plus the task
name, instruction and time. Key names follow LeRobot's simulation
environments, so LeRobot's evaluator maps them to `observation.state` and
`observation.images.<camera>` (sim/lerobot_plugin).
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).parent / "scenes" / "dinner_table.xml"
ITEMS_DIR = Path(__file__).parent / "assets" / "objects" / "ycb"
CONTROL_HZ = 50
TABLE_TOP = 0.315
ITEM_PREFIX = "ycb_"


def item_model_path(item: str) -> Path:
    """sim/assets/objects/ycb/<id>_<item>/model.xml for an item name such as 'tomato_soup_can'."""
    matches = [p for p in ITEMS_DIR.glob(f"*_{item}/model.xml")]
    if len(matches) != 1:
        raise KeyError(f"no unique YCB model for item {item!r}: {matches}")
    return matches[0]


def build_model(items: tuple[str, ...], scene: Path = SCENE) -> mujoco.MjModel:
    """The scene plus the given items, each attached once with prefix 'ycb_'.

    Items start at the origin; `reset` places them. The scene's keyframes list
    only bins and arms, so their item entries are padded (with the origin) —
    always set item poses after loading a key.
    """
    spec = mujoco.MjSpec.from_file(str(scene))
    for item in items:
        child = mujoco.MjSpec.from_file(str(item_model_path(item)))
        spec.worldbody.add_frame().attach_body(child.body(item), ITEM_PREFIX, "")
    with warnings.catch_warnings():
        # Expected: the attached arm's own solver settings lose to the scene's (design-decisions D3).
        warnings.filterwarnings("ignore", message="Attach conflict when attaching 'so101'")
        return spec.compile()


def resting_height(model: mujoco.MjModel, item: str) -> float:
    """Height of the item's body origin above a surface it stands on (its visual mesh's lowest point)."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    g = model.geom(f"{ITEM_PREFIX}{item}_visual").id
    mid = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return float(d.xpos[model.geom_bodyid[g]][2] - world[:, 2].min())


@dataclass
class Task:
    instruction: str
    items: list[str]                                   # body names (without the ycb_ prefix) on the table
    place: Callable[["WasteSortEnv", np.random.Generator], None]
    success: Callable[["WasteSortEnv"], bool]
    max_seconds: float = 25.0
    extra: dict = field(default_factory=dict)


# --- task T0: arm A lifts the can, arm B opens the metal bin, arm A puts the can in -------------
# T0 is the simplest case of T1 (README): the same task with only the can on the table, used as
# the pipeline smoke test (design-decisions D30). T1 itself (with other items) is not built yet.

# Where arm A can grasp the can: gripper vertical, sliding in from the side (debug-log #13, #28).
CAN_PATCH = dict(x=(-0.16, -0.06), y=(-0.10, -0.05))


def _place_can(env: "WasteSortEnv", rng: np.random.Generator) -> None:
    for _ in range(100):
        x, y = rng.uniform(*CAN_PATCH["x"]), rng.uniform(*CAN_PATCH["y"])
        yaw = rng.uniform(-np.pi, np.pi)
        env.set_item_pose("tomato_soup_can", [x, y, env.spawn_z("tomato_soup_can")], yaw)
        mujoco.mj_forward(env.model, env.data)
        if not env.item_touches_arm("tomato_soup_can"):
            return
    raise RuntimeError("could not place the can clear of the arms")


def _can_in_metal_bin(env: "WasteSortEnv") -> bool:
    return env.inside_site_box(env.item_pos("tomato_soup_can"), "metal_bin_interior")


TASKS: dict[str, Task] = {
    "t0_can_to_metal_bin": Task(
        instruction="Pick up the tomato soup can with arm A, open the metal bin with arm B, "
                    "and put the can in the metal bin with arm A.",
        items=["tomato_soup_can"],
        place=_place_can,
        success=_can_in_metal_bin,
    ),
}


class WasteSortEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, task: str = "t0_can_to_metal_bin", cameras: tuple[str, ...] = (),
                 image_size: tuple[int, int] = (224, 224), render_size: tuple[int, int] = (480, 640),
                 scene: Path = SCENE):
        self.task_name, self.task = task, TASKS[task]
        self.scene = scene
        self._models: dict[tuple[str, ...], tuple[mujoco.MjModel, dict[str, float]]] = {}
        self._use_items(tuple(self.task.items))
        self.n_substeps = round(1.0 / CONTROL_HZ / self.model.opt.timestep)
        self.max_steps = int(self.task.max_seconds * CONTROL_HZ)
        self.cameras, self.image_size, self.render_size = tuple(cameras), image_size, render_size
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        self.action_space = gym.spaces.Box(lo.astype(np.float32), hi.astype(np.float32))
        obs = {"agent_pos": gym.spaces.Box(-np.inf, np.inf, (self.model.nu,), np.float32)}
        if self.cameras:
            obs["pixels"] = gym.spaces.Dict({c: gym.spaces.Box(0, 255, (*image_size, 3), np.uint8) for c in self.cameras})
        self.observation_space = gym.spaces.Dict(obs)
        self.steps = 0

    @property
    def task_description(self) -> str:
        """The language instruction; LeRobot's evaluator reads it under this name."""
        return self.task.instruction

    def _use_items(self, items: tuple[str, ...]) -> None:
        """Switch to the model containing exactly `items` (built once, then cached)."""
        if items not in self._models:
            model = build_model(items, self.scene)
            self._models[items] = (model, {i: resting_height(model, i) for i in items})
        model, heights = self._models[items]
        if getattr(self, "model", None) is model:
            return
        self.items, self.model, self._rest_h = items, model, heights
        self.data = mujoco.MjData(model)
        self.home_key = model.key("home").id
        self.arm_qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        for r in getattr(self, "_renderers", {}).values():
            r.close()
        self._renderers = {}

    # --- helpers for tasks and experts (privileged sim state) ---------------------------------

    def _free_adr(self, item: str) -> int:
        return self.model.jnt_qposadr[self.model.joint(f"{ITEM_PREFIX}{item}_free").id]

    def item_pos(self, item: str) -> np.ndarray:
        return self.data.body(f"{ITEM_PREFIX}{item}").xpos.copy()

    def spawn_z(self, item: str) -> float:
        """Height of the item's origin when standing upright on the table (+2 mm to settle)."""
        return TABLE_TOP + self._rest_h[item] + 0.002

    def set_item_pose(self, item: str, pos, yaw: float = 0.0) -> None:
        a = self._free_adr(item)
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[self.model.jnt_dofadr[self.model.joint(f"{ITEM_PREFIX}{item}_free").id]:][:6] = 0

    def item_touches_arm(self, item: str) -> bool:
        b = self.model.body(f"{ITEM_PREFIX}{item}").id
        for c in self.data.contact[: self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            other = b2 if b1 == b else b1 if b2 == b else None
            if other is not None and self.model.body(other).name.startswith(("left_", "right_")):
                return True
        return False

    def inside_site_box(self, p: np.ndarray, site: str) -> bool:
        s = self.data.site(site)
        local = s.xmat.reshape(3, 3).T @ (p - s.xpos)
        return bool(np.all(np.abs(local) <= self.model.site(site).size))

    def bin_opening(self, b: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{b}_bin_slide").id]])

    # --- gym API ------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._use_items(tuple(self.task.items))
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home_key)  # bins and arms; item entries are padded
        self.task.place(self, self.np_random)
        mujoco.mj_forward(m, d)
        for _ in range(int(0.2 / m.opt.timestep)):  # let placed items settle; not part of the episode
            mujoco.mj_step(m, d)
        d.time = 0.0
        self.steps = 0
        return self._obs(), self._info()

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        success = self.task.success(self)
        terminated = success
        truncated = self.steps >= self.max_steps
        return self._obs(), float(success), terminated, truncated, self._info(success)

    def render(self, camera: str = "overview") -> np.ndarray:
        return self._render(camera, self.render_size)

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # --- internals ----------------------------------------------------------------------------

    def _render(self, camera: str, size: tuple[int, int]) -> np.ndarray:
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, *size)
        r = self._renderers[size]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def _obs(self) -> dict:
        obs = {"agent_pos": self.data.qpos[self.arm_qadr].astype(np.float32)}
        if self.cameras:
            obs["pixels"] = {c: self._render(c, self.image_size) for c in self.cameras}
        return obs

    def _info(self, success: bool = False) -> dict:
        return {"task": self.task_name, "instruction": self.task.instruction, "is_success": success,
                "time": self.steps / CONTROL_HZ}

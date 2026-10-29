"""Per-episode randomisation (the Randomisation section of 1tasks_scene.md).

`Randomization` is the one place every factor is configured. Each can be switched off, so factors
can be added and validated one at a time. Where each is drawn:

    item start positions  each task's start state: the start areas in sim/tasks.py, drawn in the
                          run_* functions in sim/skills.py (from default_rng(seed))
    start_noise   arm start pose: Gaussian noise on all 5 joints of both arms, drawn in Sim.reset
    demo_noise    demonstration noise on free-space moves, applied every step in Sim (sim/scene.py)
    the rest      drawn here, in `apply`, from their own stream (`episode_rng`), so switching them on
                  or off doesn't change an episode's item starts or noise:

    colours      item colours from a palette (held-out colours for the eval split)
    distractors  0..n round objects in zones clear of the arms' work (nothing cube-like)
    cube_yaw     cube rotation about z: ±45° on the table (the cube repeats every 90°, so this is
                 every orientation), ±20° in the drawer (arm B found no grasp from +25°)
    lighting     direction, intensity and tint of the main light; headlight level
    wording      command paraphrases (one held-out phrasing per task for the eval split)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.scene import DEMO_NOISE, DISTRACTORS, START_NOISE, TABLE_Z, Sim
from sim.tasks import yaw_quat

# Every colour stands out from everything around the cube (median rendered RGB distance >= 0.45 from
# the arms' yellow parts and black motors, the table, drawer, handle and cabinet, measured on A0
# frames; 2data_collection.md, "Cube colours"). Yellow/orange (as the arms), black (as the handle and
# motors), pink, lime and white failed. Train and eval colours don't overlap, and each eval colour
# sits between training colours (>= 0.29 RGB distance from the nearest), so eval tests unseen colours,
# not an unseen hue family. 6 and 4 colours, as before, so every seed draws the same start.
TRAIN_COLOURS = {
    "red": (0.85, 0.20, 0.15),
    "green": (0.20, 0.70, 0.25),
    "blue": (0.20, 0.45, 0.85),
    "violet": (0.35, 0.15, 0.60),
    "teal": (0.00, 0.50, 0.50),
    "crimson": (0.75, 0.05, 0.30),
}
EVAL_COLOURS = {
    "magenta": (0.85, 0.10, 0.65),
    "cyan": (0.15, 0.80, 0.85),
    "navy": (0.10, 0.15, 0.45),
    "maroon": (0.50, 0.05, 0.10),
}

# Distractor zones (x range, y range of the centre), world frame: beside the cabinet and in the
# front corners, away from the cube's table area, the cube targets, the drawer's path and the paths
# the arms take to them. One distractor per zone, so they never overlap.
DISTRACTOR_ZONES = (
    ((-0.27, -0.15), (0.10, 0.21)),  # back left
    ((0.15, 0.27), (0.10, 0.21)),  # back right
    ((-0.27, -0.18), (-0.22, -0.13)),  # front left
    ((0.18, 0.27), (-0.22, -0.13)),  # front right
)
DISTRACTOR_MOVED = 0.005  # m; a distractor pushed farther than this fails the episode

LIGHT_TILT = 20  # degrees, main light direction from vertical
LIGHT_DIFFUSE = (0.5, 0.9)  # main light intensity (scene default 0.7)
LIGHT_TINT = 0.1  # ± per colour channel (warm/cool)
HEADLIGHT_DIFFUSE = (0.4, 0.7)  # scene default 0.6
HEADLIGHT_AMBIENT = (0.2, 0.4)  # scene default 0.3

# The 20 variants (task, arm, target): one exact command each. arm is the opener in T1 and the placer in T2;
# target is None except in S4 and T1. Same order as the one-hot variant input (1tasks_scene.md).
TARGETS = ("left", "middle", "right")
VARIANTS = tuple((task, arm, target) for task in ("S1", "S2", "S3", "S4", "T1", "T2") for arm in ("A", "B")
                 for target in (TARGETS if task in ("S4", "T1") else (None,)))
# The variant input for ACT/DP (1tasks_scene.md, "Variant input"): one-hot task, arm (the opener in T1, the
# placer in T2) and target (all zero for tasks without one), as Meta-World MT10/MT50 adds a task ID.
VARIANT_DIMS = ("S1", "S2", "S3", "S4", "T1", "T2", "A", "B") + TARGETS


def variant_vector(task: str, arm: str, target: str | None = None) -> np.ndarray:
    v = np.zeros(len(VARIANT_DIMS), np.float32)
    for x in (task, arm, target):
        if x is not None:
            v[VARIANT_DIMS.index(x)] = 1.0
    return v


# Episode seeds. A seed fixes the whole start (a run_* draws it from the seed alone), so the same seed gives the
# same start in every task that shares the scene. Training: every episode gets its own seed, so every recorded
# start is different (as LIBERO's per-task init states and RoboTwin's per-episode draws); variant k records from
# its own block of TRAIN_BLOCK seeds, room for failed episodes and for picking S2's two start kinds.
# Evaluation: every variant uses the same EVAL_SEEDS, so swap tests compare commands on identical starts.
TRAIN_BLOCK = 500
TRAIN_SEEDS = range(1, 1 + TRAIN_BLOCK * len(VARIANTS))  # 1-10,000
EVAL_SEEDS = range(100_000, 101_000)


def train_seeds(task: str, arm: str, target: str | None = None) -> range:
    """The training seeds of one variant: its own block, shared with no other variant."""
    k = VARIANTS.index((task, arm, target))
    return range(TRAIN_SEEDS.start + k * TRAIN_BLOCK, TRAIN_SEEDS.start + (k + 1) * TRAIN_BLOCK)

# Command paraphrases per task. The first is the canonical command (1tasks_scene.md); the last is
# held out for the eval split.
COMMANDS = {
    "S1": ("open the drawer with arm {arm}",
           "use arm {arm} to open the drawer",
           "arm {arm}, pull the drawer open"),
    "S2": ("close the drawer with arm {arm}",
           "use arm {arm} to close the drawer",
           "arm {arm}, push the drawer shut"),
    "S3": ("put the cube in the drawer with arm {arm}",
           "use arm {arm} to put the cube in the drawer",
           "arm {arm}, place the cube inside the drawer"),
    "S4": ("take the cube out of the drawer and put it on the {target} of the table with arm {arm}",
           "use arm {arm} to take the cube out of the drawer and put it on the {target} of the table",
           "arm {arm}, move the cube from the drawer to the {target} of the table"),
    "T1": ("open the drawer with arm {opener}, then take the cube out and put it on the {target} of the table with arm {picker}",
           "use arm {opener} to open the drawer, then arm {picker} to take the cube out and put it on the {target} of the table",
           "arm {opener} opens the drawer and arm {picker} moves the cube to the {target} of the table"),
    "T2": ("put the cube in the drawer with arm {placer}, then close the drawer with arm {closer}",
           "use arm {placer} to put the cube in the drawer, then arm {closer} to close it",
           "arm {placer} places the cube in the drawer and arm {closer} pushes the drawer shut"),
}


@dataclass(frozen=True)
class Randomization:
    start_noise: float = START_NOISE  # rad, σ per arm joint at reset
    demo_noise: float = DEMO_NOISE  # rad, σ of the demonstration noise
    colours: bool = True
    distractors: int = 2  # at most this many (0 = none)
    cube_yaw: float = 45.0  # degrees, ± range on the table
    cube_yaw_drawer: float = 20.0  # degrees, ± range in the drawer
    lighting: bool = True
    wording: bool = True
    split: str = "train"  # "train" or "eval": eval uses the held-out colours and phrasings
    # One factor held out on its own (the evaluation conditions in 1tasks_scene.md): "train" or "eval"
    # overrides `split` for that factor; None follows `split`.
    colours_split: str | None = None
    wording_split: str | None = None

    def split_of(self, factor: str) -> str:
        """The split one held-out factor uses ("colours" or "wording")."""
        return getattr(self, f"{factor}_split") or self.split


NONE = Randomization(start_noise=0.0, demo_noise=0.0, colours=False, distractors=0, cube_yaw=0.0, cube_yaw_drawer=0.0,
                     lighting=False, wording=False)


def episode_rng(seed: int) -> np.random.Generator:
    """The randomisation stream, separate from the task's start-position stream."""
    return np.random.default_rng([seed, 1])


def apply(sim: Sim, seed: int, cfg: Randomization) -> dict:
    """Randomise the scene for one episode (after `sim.clear_items()`, before the task places
    its items). Returns what was drawn, for the episode record."""
    rng = episode_rng(seed)
    info: dict = {}
    palette = TRAIN_COLOURS if cfg.split_of("colours") == "train" else EVAL_COLOURS
    names = list(palette)
    if cfg.colours:
        cube = names[rng.integers(len(names))]
        set_colour(sim, "cube", palette[cube])
        info.update(cube_colour=cube)
    if cfg.cube_yaw:
        # One draw scaled to each range; the task uses whichever applies to where the cube starts.
        u = rng.uniform(-1, 1)
        sim.cube_yaw = float(np.deg2rad(u * cfg.cube_yaw))
        sim.cube_yaw_drawer = float(np.deg2rad(u * cfg.cube_yaw_drawer))
        info["cube_yaw"] = round(float(u * cfg.cube_yaw), 1)
        info["cube_yaw_drawer"] = round(float(u * cfg.cube_yaw_drawer), 1)
    if cfg.distractors:
        n = int(rng.integers(0, cfg.distractors + 1))
        bodies = rng.choice(DISTRACTORS, n, replace=False)
        zones = rng.choice(len(DISTRACTOR_ZONES), n, replace=False)
        # Distractor colours come from the full palette but never match the cube's, so the
        # command's object is the only one of its colour.
        free = [c for c in names if c != info.get("cube_colour")]
        placed = []
        for body, zone in zip(map(str, bodies), zones):
            (x0, x1), (y0, y1) = DISTRACTOR_ZONES[zone]
            xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            half_h = distractor_half_height(sim, body)
            sim.place(body, (xy[0], xy[1], TABLE_Z + half_h + 0.001), quat=yaw_quat(rng.uniform(0, np.pi)))
            colour = free[rng.integers(len(free))]
            set_colour(sim, body, palette[colour])
            sim.distractor_xy[body] = xy
            placed.append(dict(body=body, xy=np.round(xy, 3).tolist(), colour=colour))
        info["distractors"] = placed
    if cfg.lighting:
        info["lighting"] = randomize_lighting(sim, rng)
    return info


def set_colour(sim: Sim, body: str, rgb):
    sim.model.geom_rgba[sim.model.geom(body).id, :3] = rgb


def distractor_half_height(sim: Sim, body: str) -> float:
    g = sim.model.geom(body)
    # cylinder: size = (radius, half-height, -); ellipsoid: (a, b, c)
    return float(g.size[1] if g.type[0] == 5 else g.size[2])


def randomize_lighting(sim: Sim, rng: np.random.Generator) -> dict:
    m = sim.model
    tilt, azimuth = np.deg2rad(rng.uniform(0, LIGHT_TILT)), rng.uniform(0, 2 * np.pi)
    m.light_dir[0] = (np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), -np.cos(tilt))
    level = rng.uniform(*LIGHT_DIFFUSE)
    m.light_diffuse[0] = level * (1 + rng.uniform(-LIGHT_TINT, LIGHT_TINT, 3))
    m.vis.headlight.diffuse[:] = rng.uniform(*HEADLIGHT_DIFFUSE)
    m.vis.headlight.ambient[:] = rng.uniform(*HEADLIGHT_AMBIENT)
    return dict(tilt=round(float(np.rad2deg(tilt)), 1), level=round(float(level), 2))


def distractors_moved(sim: Sim) -> float:
    """Largest horizontal distance any distractor moved from where it was placed (m)."""
    return max((float(np.linalg.norm(sim.body_pos(b)[:2] - xy)) for b, xy in sim.distractor_xy.items()), default=0.0)


def command(task: str, seed: int, cfg: Randomization, **names) -> str:
    """The episode's command string. `names`: arm / target / opener / picker / placer / closer."""
    options = COMMANDS[task]
    if not cfg.wording:
        template = options[0]
    elif cfg.split_of("wording") == "train":
        template = options[int(np.random.default_rng([seed, 2]).integers(len(options) - 1))]
    else:
        template = options[-1]
    return template.format(**names)

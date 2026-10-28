"""What every episode needs whoever acts, the scripted expert (sim/skills.py) or a learned policy: the
variant's start state, monitors that run every physics step, and the task's success check
(1tasks_scene.md). Nothing here reads the expert's plan, so a policy rollout is scored exactly as a
demonstration is, as LIBERO's and ManiSkill's success checks read only the simulator.

    sim, start = reset(("S4", "A", "left"), seed, rand)   # start scene drawn from the seed, settled
    ...                                                    # act: set sim.data.ctrl, sim.step(CONTROL_EVERY)
    ok, checks = success(sim, start)                       # at the end of the episode

Besides the task's own condition, every task checks who did it: in a primitive the named arm
touched the item it works on (drawer or cube) and the other arm never left home; in a composite each
arm touched its own item and the first arm never touched the second step's. Otherwise a policy that
opens the drawer with arm B when told arm A would pass.
"""

from __future__ import annotations

import numpy as np

from sim import randomize as RZ
from sim import tasks as T
from sim.randomize import Randomization
from sim.scene import ARM_NAME, ARMS, CONTROL_EVERY, Sim

PUSHED = 0.005  # m; an item (or the drawer) moved farther than this when it should stay put fails the episode
IDLE_MOVE = 0.1  # rad; an arm with a joint farther than this from its start has left home (the continue rate)
DROPPED = 0.015  # m; the cube falling farther than this with no arm touching it was dropped, not set down
S2_CUBE_IN_DRAWER = 0.5  # share of S2 episodes that start exactly like S4 (see reset)
ITEM_BODIES = ("drawer", "cube")  # what the monitor tracks contacts with (the drawer body includes its handle)
PRIMITIVE_ITEM = {"S1": "drawer", "S2": "drawer", "S3": "cube", "S4": "cube"}


def new_sim(seed: int, rand: Randomization) -> tuple[Sim, dict]:
    """A fresh Sim with the episode's randomisation applied; returns it and what was drawn."""
    sim = Sim(seed=seed, noise=rand.start_noise)
    sim.demo_noise = rand.demo_noise
    sim.control_every = CONTROL_EVERY
    sim.clear_items()
    return sim, RZ.apply(sim, seed, rand)


def other(arm: str) -> str:
    return "B" if arm == "A" else "A"


def roles(variant) -> dict:
    """The command's names for one variant: arm (and target), or opener/picker (T1), placer/closer (T2)."""
    task, arm, target = variant
    if task == "T1":
        return dict(opener=arm, picker=other(arm), target=target)
    if task == "T2":
        return dict(placer=arm, closer=other(arm))
    return dict(arm=arm) if target is None else dict(arm=arm, target=target)


def partly_open_start(sim: Sim, rng) -> float:
    """Drawer partly open (DRAWER_PARTLY_OPEN), cube on the table: the shared start of S2, S3 and
    T2. Returns the opening."""
    sim.set_drawer(float(rng.uniform(*T.DRAWER_PARTLY_OPEN)))
    T.place_cube_on_table(sim, T.table_xy(rng))
    return sim.drawer_opening()


def cube_in_drawer_frame(sim: Sim) -> np.ndarray:
    """Cube x, y in the drawer's frame (moves with the drawer)."""
    return sim.body_pos("cube")[:2] + sim.drawer_opening() * T.Y[:2]


def reset(variant, seed: int, rand: Randomization = Randomization()) -> tuple[Sim, dict]:
    """The variant's start scene, drawn from `seed` alone (so variants sharing a scene get the same
    start for the same seed), with a fresh Monitor attached. Returns the Sim and `start`: what the
    success check compares against, plus the episode's command and what the randomisation drew.

    Three start scenes (1tasks_scene.md): drawer closed with the cube inside (S1, T1); drawer partly
    open with the cube on the table (S3, T2, and half of S2); drawer fully open with the cube inside
    (S4, and the other half of S2)."""
    task = variant[0]
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    start = dict(variant=tuple(variant), seed=seed, scene=scene)
    if task == "S2":
        start["cube_in_drawer"] = bool(rng.random() < S2_CUBE_IN_DRAWER)
    if task in ("S1", "T1"):
        T.place_cube_in_drawer(sim, T.drawer_xy(rng, 0.0))
        sim.step(50)
    elif task == "S4" or start.get("cube_in_drawer"):
        sim.set_drawer(T.DRAWER_OPEN)
        T.place_cube_in_drawer(sim, T.drawer_xy(rng, T.DRAWER_OPEN))
        sim.step(50)
    else:
        partly_open_start(sim, rng)
    start.update(opening=sim.drawer_opening(), cube_drawer_xy=cube_in_drawer_frame(sim),
                 command=RZ.command(task, seed, rand, **roles(variant)))
    sim.monitor = Monitor(sim)
    return sim, start


class Monitor:
    """Runs every physics step (Sim.step calls `update`): which items each arm touched, how far each
    arm moved from its start, and the cube's longest fall with no arm touching it."""

    def __init__(self, sim: Sim):
        m = sim.model
        self.geom_arm = np.full(m.ngeom, -1)
        for i, name in enumerate(ARMS):
            self.geom_arm[list(sim.arms[name].geoms)] = i
        self.geom_item = np.full(m.ngeom, -1)
        for i, body in enumerate(ITEM_BODIES):
            self.geom_item[m.geom_bodyid == m.body(body).id] = i
        self.cube = m.body("cube").id
        self.q0 = {a: sim.q(a) for a in ARMS}
        self.moved = {a: 0.0 for a in ARMS}  # rad, largest joint distance from the start, per arm
        self.touched = {a: set() for a in ARMS}  # items in ITEM_BODIES each arm has touched
        self.release_z = None  # cube height at the last step an arm touched it
        self.drop = 0.0  # m, the cube's longest fall since an arm last touched it

    def update(self, sim: Sim):
        d = sim.data
        g = d.contact.geom[: d.ncon]
        arm, item = self.geom_arm[g], self.geom_item[g]
        cube_touched = False
        for a, i in ((0, 1), (1, 0)):
            hit = (arm[:, a] >= 0) & (item[:, i] >= 0)
            for ai, ii in zip(arm[hit, a], item[hit, i]):
                self.touched[ARMS[ai]].add(ITEM_BODIES[ii])
                cube_touched |= ITEM_BODIES[ii] == "cube"
        for a in ARMS:
            self.moved[a] = max(self.moved[a], float(np.abs(d.qpos[sim.arms[a].qadr] - self.q0[a]).max()))
        z = float(d.xpos[self.cube, 2])
        if cube_touched:
            self.release_z = z
        elif self.release_z is not None:
            self.drop = max(self.drop, self.release_z - z)


def cube_upright_in_drawer(sim: Sim) -> bool:
    """The cube rests on the drawer floor (not on a wall or the front), upright."""
    on_floor = abs(sim.body_pos("cube")[2] - (T.DRAWER_FLOOR_Z + T.CUBE_H / 2)) < 0.005
    return bool(on_floor and sim.data.body("cube").xmat.reshape(3, 3)[2, 2] > 0.95)


def cube_on_table_at(sim: Sim, xy, tol: float = 0.02) -> bool:
    p = sim.body_pos("cube")
    upright = abs(sim.data.body("cube").xmat.reshape(3, 3)[2, 2]) > 0.95
    on_table = abs(p[2] - (T.TABLE_Z + T.CUBE_H / 2)) < 0.005
    return bool(np.linalg.norm(p[:2] - np.asarray(xy)) < tol and upright and on_table)


def success(sim: Sim, start: dict) -> tuple[bool, dict]:
    """The task's success check (1tasks_scene.md) at the end of an episode, from the simulator state,
    `start` and the Monitor alone. Returns whether every check passed, and each check by name."""
    task, arm, target = start["variant"]
    m = sim.monitor
    opening = sim.drawer_opening()
    first, second = ARM_NAME[arm], ARM_NAME[other(arm)]
    c = {}
    if task == "S1":
        c["drawer_open"] = opening > T.DRAWER_OPEN - 0.01
        c["cube_kept_place"] = float(np.linalg.norm(cube_in_drawer_frame(sim) - start["cube_drawer_xy"])) < PUSHED
        c["cube_upright_in_drawer"] = cube_upright_in_drawer(sim)
    elif task == "S2":
        c["drawer_closed"] = opening < 0.01
        if start["cube_in_drawer"]:
            c["cube_upright_in_drawer"] = cube_upright_in_drawer(sim)
    elif task in ("S3", "T2"):
        c["cube_upright_in_drawer"] = cube_upright_in_drawer(sim)
        c["set_down"] = m.drop < DROPPED
        if task == "S3":
            c["drawer_still"] = abs(opening - start["opening"]) < PUSHED
        else:
            c["drawer_closed"] = opening < 0.01
    else:  # S4, T1
        c["cube_at_target"] = cube_on_table_at(sim, T.CUBE_TARGETS[target])
    if task in PRIMITIVE_ITEM:
        c["named_arm_acted"] = PRIMITIVE_ITEM[task] in m.touched[first]
        c["other_arm_stayed_home"] = m.moved[second] < IDLE_MOVE
    else:  # T1: first arm opens, second takes the cube out; T2: first puts the cube in, second closes
        item1, item2 = ("drawer", "cube") if task == "T1" else ("cube", "drawer")
        c["first_arm_acted"] = item1 in m.touched[first]
        c["second_arm_acted"] = item2 in m.touched[second]
        c["first_arm_kept_to_its_step"] = item2 not in m.touched[first]
    c["no_arm_contact"] = sim.arm_contacts == 0
    c["distractors_still"] = RZ.distractors_moved(sim) < RZ.DISTRACTOR_MOVED
    c = {k: bool(v) for k, v in c.items()}
    return all(c.values()), c

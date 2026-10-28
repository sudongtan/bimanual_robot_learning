"""Scripted skills for the tasks in 1tasks_scene.md, built from the patterns validated in deprecated/sim/expert.py:
joint-space moves to a hover point (with demonstration noise), straight lines for precise moves
(no noise), one fixed grasp per target, and every skill starts and ends with the arm at home.

Each `run_*` function plays a whole task in a fresh Sim and returns a result dict whose
`success` field is the task's success check.
"""

from __future__ import annotations

import numpy as np

from sim import randomize as RZ
from sim import tasks as T
from sim.randomize import Randomization
from sim.scene import ARM_NAME, CONTROL_EVERY, GRIPPER_CLOSED, TABLE_Z, Sim, solve_ik

HOVER = 0.05  # m above the pre-grasp, reached in joint space
CARRY_Z = 0.08  # m, fingertip height above the tabletop while carrying the cube
PUSHED = 0.005  # m; an item pushed farther than this while being approached fails the episode
PULL_PAST = 0.01  # m past the drawer's end stop that a full opening aims for (see move_drawer)
HOLD_S = 2.0  # s, recorded hold with both arms at home at the end of every task: one ACT action chunk
# (100 frames at 50 fps), so a policy is trained on staying still after the task (stop tests, 1tasks_scene.md)
HOLD_STILL = 0.05  # rad; a joint moving farther than this during the hold fails the episode (same as the stop test)


# --- building blocks -----------------------------------------------------------


def go_home(sim: Sim, arm: str, rec=None):
    """Lift clear, then a joint-space move home."""
    sim.move_line(arm, sim.tip(arm) + 0.04 * T.Z, 0.4, on_step=rec)
    with sim.noisy(arm, rec):
        sim.move_joints({arm: sim.home}, 1.5, on_step=rec)


def approach(sim: Sim, arm: str, plan: T.GraspPlan, grip: float, rec=None, follow_plan: bool = False):
    """Open the gripper, go to a hover point above the pre-grasp, then straight in to the grasp.
    `follow_plan`: reach the hover point in the plan's own arm posture (e.g. its chosen wrist
    roll) instead of the IK answer closest to the current joints."""
    sim.data.ctrl[sim.arms[arm].grip_act] = grip
    with sim.noisy(arm, rec):
        if follow_plan:
            a = sim.arms[arm]
            hover = solve_ik(sim.model, a, plan.pre + HOVER * T.Z, point=plan.point, jaw_perp=plan.jaw_perp, q0=plan.q_pre)
            sim.move_joints({arm: hover.q}, 1.6, on_step=rec)
        else:
            sim.move_to(arm, plan.pre + HOVER * T.Z, 1.6, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    sim.move_line(arm, plan.pre, 0.5, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    sim.move_line(arm, plan.grasp, 0.6, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)


def approach_item(sim: Sim, arm: str, plan: T.GraspPlan, grip: float, body: str, rec=None, follow_plan: bool = False):
    """`approach` for an item resting somewhere; records how far it was pushed before the grasp."""
    start = sim.body_pos(body)[:2]
    approach(sim, arm, plan, grip, rec, follow_plan)
    sim.pushed = max(sim.pushed, float(np.linalg.norm(sim.body_pos(body)[:2] - start)))


def move_drawer(sim: Sim, arm: str, target_opening: float, rec=None) -> bool:
    """Grasp the handle, slide the drawer to `target_opening`, release, go home."""
    plan = T.plan_handle_grasp(sim, arm)
    if plan is None:
        return False
    approach(sim, arm, plan, T.HANDLE_OPEN, rec)
    # Close gradually: the fixed jaw stops JAW_CLEARANCE off the bar, so closing drags the drawer ~5 mm, and a
    # sudden close jerks it out from under the cube (4 mm slip in S1/T1); ramped, the cube rides along.
    sim.grip(arm, GRIPPER_CLOSED, 0.4, rec, ramp=True)
    # The arm lags its target against the drawer's friction and stops ~6 mm short, so to open fully
    # it aims past the end stop: the drawer then ends where S2 and S4 start (13.9-14 cm, not 13.4).
    aim = target_opening + PULL_PAST if target_opening >= T.DRAWER_OPEN else target_opening
    delta = sim.drawer_opening() - aim  # handle moves +y when the drawer closes
    sim.move_line(arm, sim.tip(arm) + delta * T.Y, 1.0 + 8 * abs(delta), point=plan.point, jaw_perp=T.X, on_step=rec)
    sim.grip(arm, T.HANDLE_OPEN, 0.3, rec)
    sim.move_line(arm, sim.tip(arm) - 0.05 * sim.pointing(arm), 0.4, on_step=rec)  # back off the bar
    go_home(sim, arm, rec)
    return True


def pick_cube(sim: Sim, arm: str, rec=None, grip_height: float = 0.02, jaw_perp=None) -> bool:
    """Grasp the cube where it is and lift it to carry height."""
    plan = T.plan_cube_grasp(sim, arm, grip_height=grip_height, jaw_perp=jaw_perp)
    if plan is None:
        return False
    # Keep the plan's wrist roll: the jaw constraint allows either roll, and the other one puts the
    # off-centre fixed jaw on the wrong side, onto a rotated cube.
    approach_item(sim, arm, plan, T.cube_open(sim), "cube", rec, follow_plan=True)
    sim.grip(arm, GRIPPER_CLOSED, 0.5, rec)
    sim.move_line(arm, sim.tip(arm) + 0.03 * T.Z, 0.3, point=plan.point, on_step=rec)  # clear the surface
    with sim.noisy(arm, rec):
        lift = sim.tip(arm).copy()
        lift[2] = max(lift[2], TABLE_Z + CARRY_Z)
        sim.move_line(arm, lift, 0.5, point=plan.point, on_step=rec)
    return holding(sim, arm)


def place_cube(sim: Sim, arm: str, xy, rec=None, point=None):
    """Carry the held cube above `xy`, lower it onto the table, release, go home. `point`: gripper
    pointing while placing (default: the standard grasp tilt); keep the grasp's own angle so the
    cube isn't rotated before it lands."""
    above = np.array([xy[0], xy[1], TABLE_Z + CARRY_Z])
    if point is None:
        point = T.tilted_down(T.radial(sim, arm, above), T.CUBE_TILT)
    with sim.noisy(arm, rec):
        sim.move_line(arm, above, 1.2, point=point, waypoints=30, on_step=rec)
    # Lower until the cube's bottom is ~5 mm above the table, then let go.
    drop = sim.tip(arm) - (sim.body_pos("cube")[2] - T.CUBE_H / 2 - TABLE_Z - 0.005) * T.Z
    sim.move_line(arm, drop, 0.6, point=point, on_step=rec)
    sim.grip(arm, T.CUBE_OPEN, 0.4, rec)
    sim.move_line(arm, sim.tip(arm) - 0.05 * point, 0.4, on_step=rec)
    go_home(sim, arm, rec)


def place_cube_in_drawer(sim: Sim, arm: str, rec=None) -> bool:
    """Carry the held cube into the open drawer (T.DRAWER_DROP), set it down, release, go home:
    S4's in-drawer approach in reverse, ending with the cube where S4 would pick it up."""
    plan = T.plan_drawer_drop(sim, arm)
    if plan is None:
        return False
    a = sim.arms[arm]
    with sim.noisy(arm, rec):
        hover = solve_ik(sim.model, a, plan.pre + HOVER * T.Z, point=plan.point, jaw_perp=plan.jaw_perp, q0=plan.q_pre)
        sim.move_joints({arm: hover.q}, 1.6, on_step=rec)
    sim.move_line(arm, plan.pre, 0.5, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    sim.move_line(arm, plan.grasp + 0.03 * T.Z, 0.5, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    # Lower until the cube's bottom is ~5 mm above the drawer floor (measured, as in place_cube),
    # then let go.
    drop = sim.tip(arm) - (sim.body_pos("cube")[2] - T.CUBE_H / 2 - T.DRAWER_FLOOR_Z - 0.005) * T.Z
    sim.move_line(arm, drop, 0.4, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    # Released higher than this, the arm was blocked on the way down and the cube is dropped,
    # not set down (it can still land upright by luck, so this is checked separately).
    set_down = sim.body_pos("cube")[2] - T.CUBE_H / 2 - T.DRAWER_FLOOR_Z < 0.015
    sim.grip(arm, T.CUBE_OPEN_DRAWER, 0.4, rec)
    # Back out along the approach, keeping the drop pose so the open jaws don't swing into the
    # drawer walls.
    sim.move_line(arm, sim.tip(arm) - 0.05 * plan.point, 0.4, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    go_home(sim, arm, rec)
    return bool(set_down)


def holding(sim: Sim, arm: str) -> bool:
    return bool(sim.touching(arm, "cube")) and sim.body_pos("cube")[2] > TABLE_Z + T.CUBE_H / 2 + 0.01


def cube_on_table_at(sim: Sim, xy, tol: float = 0.02) -> bool:
    p = sim.body_pos("cube")
    upright = abs(sim.data.body("cube").xmat.reshape(3, 3)[2, 2]) > 0.95
    on_table = abs(p[2] - (TABLE_Z + T.CUBE_H / 2)) < 0.005
    return bool(np.linalg.norm(p[:2] - np.asarray(xy)) < tol and upright and on_table)


def new_sim(seed: int, rand: Randomization) -> tuple[Sim, dict]:
    """A fresh Sim with the episode's randomisation applied; returns it and what was drawn."""
    sim = Sim(seed=seed, noise=rand.start_noise)
    sim.demo_noise = rand.demo_noise
    sim.control_every = CONTROL_EVERY
    sim.clear_items()
    return sim, RZ.apply(sim, seed, rand)


def hold(sim: Sim, rec=None) -> tuple[float, float]:
    """Both arms stay at home for HOLD_S (recorded). Returns the largest joint motion (rad) and the
    largest cube or drawer motion (m) during the hold, both measured from its start."""
    q0 = {a: sim.q(a) for a in sim.arms}
    cube, opening = sim.body_pos("cube"), sim.drawer_opening()
    motion = 0.0

    def on_step(s: Sim):
        nonlocal motion
        motion = max(motion, *(float(np.abs(s.q(a) - q0[a]).max()) for a in s.arms))
        if rec is not None:
            rec(s)

    sim.step(int(HOLD_S / sim.model.opt.timestep), on_step)
    return motion, max(float(np.linalg.norm(sim.body_pos("cube") - cube)), abs(sim.drawer_opening() - opening))


def finish(sim: Sim, result: dict, scene: dict, rec=None) -> dict:
    """Record the hold, then add the checks that apply to every episode."""
    hold_motion, hold_moved = hold(sim, rec)
    moved = RZ.distractors_moved(sim)
    result.update(arm_contacts=sim.arm_contacts, max_jump=round(sim.max_jump, 2),
                  distractor_moved=round(moved, 4), pushed=round(sim.pushed, 4),
                  hold_motion=round(hold_motion, 4), hold_moved=round(hold_moved, 4), scene=scene)
    result["success"] = (bool(result.get("success")) and sim.arm_contacts == 0 and moved < RZ.DISTRACTOR_MOVED
                         and sim.pushed < PUSHED and hold_motion < HOLD_STILL and hold_moved < PUSHED)
    return result


# --- tasks (1tasks_scene.md) --------------------------------------------------------------


def cube_upright_in_drawer(sim: Sim) -> bool:
    """The cube rests on the drawer floor (not on a wall or the front), upright."""
    on_floor = abs(sim.body_pos("cube")[2] - (T.DRAWER_FLOOR_Z + T.CUBE_H / 2)) < 0.005
    return bool(on_floor and sim.data.body("cube").xmat.reshape(3, 3)[2, 2] > 0.95)


def run_S1(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S1: open the drawer with `arm` ("A" or "B"). Starts exactly like T1 (cube inside the closed
    drawer), so only the command tells S1 from T1. The cube rides out with the drawer and must
    keep its place in it (< 5 mm relative to the drawer), upright."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    T.place_cube_in_drawer(sim, T.drawer_xy(rng, 0.0))
    sim.step(50)

    def cube_xy():  # in the drawer's frame
        return sim.body_pos("cube")[:2] + sim.drawer_opening() * T.Y[:2]

    start = cube_xy()
    ok = move_drawer(sim, ARM_NAME[arm], T.DRAWER_OPEN, rec)
    moved = float(np.linalg.norm(cube_xy() - start))
    return finish(sim, dict(task="S1", arm=arm, seed=seed, opened=round(sim.drawer_opening(), 3),
                            cube_moved=round(moved, 4), command=RZ.command("S1", seed, rand, arm=arm),
                            success=ok and sim.drawer_opening() > T.DRAWER_OPEN - 0.01 and moved < 0.005
                            and cube_upright_in_drawer(sim)),
                  scene, rec)


S2_CUBE_IN_DRAWER = 0.5  # share of S2 episodes that start exactly like S4 (see run_S2)


def partly_open_start(sim: Sim, rng) -> float:
    """Drawer partly open (DRAWER_PARTLY_OPEN), cube on the table: the shared start of S2, S3 and
    T2. Returns the opening."""
    sim.set_drawer(float(rng.uniform(*T.DRAWER_PARTLY_OPEN)))
    T.place_cube_on_table(sim, T.table_xy(rng))
    return sim.drawer_opening()


def run_S2(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S2: close the drawer with `arm`. Half the episodes start like S3 and T2 (drawer partly open,
    cube on the table); the other half start exactly like S4 (drawer fully open, cube in the
    drawer). Either way only the command tells S2 from the others (as in Language-Table and
    CALVIN, where one start state carries several instructions). A cube in the drawer rides in
    and must stay upright in it."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    in_drawer = bool(rng.random() < S2_CUBE_IN_DRAWER)
    if in_drawer:
        sim.set_drawer(T.DRAWER_OPEN)
        T.place_cube_in_drawer(sim, T.drawer_xy(rng, T.DRAWER_OPEN))
        sim.step(50)
    else:
        partly_open_start(sim, rng)
    ok = move_drawer(sim, ARM_NAME[arm], 0.0, rec)
    cube_ok = not in_drawer or cube_upright_in_drawer(sim)
    return finish(sim, dict(task="S2", arm=arm, seed=seed, cube_in_drawer=in_drawer, opening=round(sim.drawer_opening(), 3),
                            command=RZ.command("S2", seed, rand, arm=arm),
                            success=ok and sim.drawer_opening() < 0.01 and cube_ok), scene, rec)


def put_cube_in_drawer(sim: Sim, arm: str, rec=None) -> tuple[bool, float]:
    """Pick the cube off the table and set it down in the (partly) open drawer. Returns whether
    the cube ended upright in the drawer, and how far the drawer moved meanwhile."""
    opening = sim.drawer_opening()
    ok = pick_cube(sim, arm, rec) and place_cube_in_drawer(sim, arm, rec)
    return ok and cube_upright_in_drawer(sim), abs(sim.drawer_opening() - opening)


def run_S3(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S3: put the cube in the drawer with `arm`. Starts like S2 and T2 (drawer partly open, cube
    anywhere on the table), so only the command tells them apart. The cube must end upright in the
    drawer, and the drawer must not move."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    partly_open_start(sim, rng)
    in_drawer, drawer_moved = put_cube_in_drawer(sim, ARM_NAME[arm], rec)
    return finish(sim, dict(task="S3", arm=arm, seed=seed, drawer_moved=round(drawer_moved, 4),
                            command=RZ.command("S3", seed, rand, arm=arm),
                            success=in_drawer and drawer_moved < PUSHED), scene, rec)


def run_S4(arm: str, target: str = "middle", seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S4: take the cube out of the open drawer and put it on a named spot (CUBE_TARGETS) with `arm`."""
    goal = T.CUBE_TARGETS[target]
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    sim.set_drawer(T.DRAWER_OPEN)
    T.place_cube_in_drawer(sim, T.drawer_xy(rng, T.DRAWER_OPEN))
    sim.step(50)
    a = ARM_NAME[arm]
    picked = pick_cube(sim, a, rec)
    if picked:
        place_cube(sim, a, goal, rec)
    return finish(sim, dict(task="S4", arm=arm, target=target, seed=seed, picked=picked,
                            command=RZ.command("S4", seed, rand, arm=arm, target=target),
                            success=picked and cube_on_table_at(sim, goal)), scene, rec)


def other(arm: str) -> str:
    return "B" if arm == "A" else "A"


def run_T1(opener: str, target: str = "middle", seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """T1 = S1 then S4: `opener` opens the drawer (cube inside), the other arm takes the cube out
    and puts it on a named spot (CUBE_TARGETS)."""
    goal = T.CUBE_TARGETS[target]
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    T.place_cube_in_drawer(sim, T.drawer_xy(rng, 0.0))
    sim.step(50)
    picker = other(opener)
    opened = move_drawer(sim, ARM_NAME[opener], T.DRAWER_OPEN, rec)
    sim.subtask = 1  # the opener is home: the S4 part (2data_collection.md, design point 5)
    sim.step(100, rec)
    picked = opened and pick_cube(sim, ARM_NAME[picker], rec)
    if picked:
        place_cube(sim, ARM_NAME[picker], goal, rec)
    return finish(sim, dict(task="T1", opener=opener, target=target, seed=seed, opened=round(sim.drawer_opening(), 3),
                            picked=picked,
                            command=RZ.command("T1", seed, rand, opener=opener, picker=picker, target=target),
                            success=picked and cube_on_table_at(sim, goal)), scene, rec)


def run_T2(placer: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """T2 = S3 then S2: `placer` puts the cube in the partly open drawer and goes home, then the
    other arm closes the drawer. Starts like S2 and S3. The cube must end upright in the closed
    drawer."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    partly_open_start(sim, rng)
    closer = other(placer)
    in_drawer, _ = put_cube_in_drawer(sim, ARM_NAME[placer], rec)
    placed = in_drawer
    sim.subtask = 1  # the placer is home: the S2 part (2data_collection.md, design point 5)
    sim.step(100, rec)
    closed = placed and move_drawer(sim, ARM_NAME[closer], 0.0, rec)
    return finish(sim, dict(task="T2", placer=placer, seed=seed, placed=placed,
                            opening=round(sim.drawer_opening(), 3),
                            command=RZ.command("T2", seed, rand, placer=placer, closer=closer),
                            success=closed and sim.drawer_opening() < 0.01 and cube_upright_in_drawer(sim)), scene, rec)

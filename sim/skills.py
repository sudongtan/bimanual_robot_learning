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
from sim.scene import ARM_NAME, GRIPPER_CLOSED, TABLE_Z, Sim, solve_ik

HOVER = 0.05  # m above the pre-grasp, reached in joint space
CARRY_Z = 0.08  # m, fingertip height above the tabletop while carrying the cube
PUSHED = 0.005  # m; an item pushed farther than this while being approached fails the episode


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
    sim.grip(arm, GRIPPER_CLOSED, 0.4, rec)
    delta = sim.drawer_opening() - target_opening  # handle moves +y when the drawer closes
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
    sim.clear_items()
    return sim, RZ.apply(sim, seed, rand)


def finish(sim: Sim, result: dict, scene: dict) -> dict:
    sim.step(250)
    moved = RZ.distractors_moved(sim)
    result.update(arm_contacts=sim.arm_contacts, max_jump=round(sim.max_jump, 2),
                  distractor_moved=round(moved, 4), pushed=round(sim.pushed, 4), scene=scene)
    result["success"] = (bool(result.get("success")) and sim.arm_contacts == 0 and moved < RZ.DISTRACTOR_MOVED
                         and sim.pushed < PUSHED)
    return result


# --- tasks (1tasks_scene.md) --------------------------------------------------------------


S1_CUBE_IN_DRAWER = 0.5  # share of S1 episodes that start exactly like T2 (see run_S1)


def run_S1(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S1: open the drawer with `arm` ("A" or "B"). Half the episodes start with the cube on the
    table as a bystander; the other half start exactly like T2 (cube inside the closed drawer), so
    only the command tells S1 from T2 (see run_S2). The cube must not move: on the table it stays
    put; in the drawer it rides out with it, so it must keep its place in the drawer, upright."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    in_drawer = bool(rng.random() < S1_CUBE_IN_DRAWER)
    if in_drawer:
        T.place_cube_in_drawer(sim, T.drawer_xy(rng, 0.0))
        sim.step(50)
    else:
        T.place_cube_on_table(sim, T.table_xy(rng))

    def cube_xy():  # in the drawer's frame if the cube rides in it
        return sim.body_pos("cube")[:2] + (in_drawer * sim.drawer_opening()) * T.Y[:2]

    start = cube_xy()
    ok = move_drawer(sim, ARM_NAME[arm], T.DRAWER_OPEN, rec)
    moved = float(np.linalg.norm(cube_xy() - start))
    upright = sim.data.body("cube").xmat.reshape(3, 3)[2, 2] > 0.95
    return finish(sim, dict(task="S1", arm=arm, seed=seed, cube_in_drawer=in_drawer, opened=round(sim.drawer_opening(), 3),
                            cube_moved=round(moved, 4), command=RZ.command("S1", seed, rand, arm=arm),
                            success=ok and sim.drawer_opening() > T.DRAWER_OPEN - 0.01 and moved < 0.005 and upright),
                  scene)


S2_CUBE_IN_DRAWER = 0.5  # share of S2 episodes that start exactly like S4 (see run_S2)


def run_S2(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S2: close the drawer with `arm`. Half the episodes start with the drawer partly open
    (DRAWER_PARTLY_OPEN) and the cube on the table, exactly like S3's drawer-open start; the other
    half start exactly like S4 (drawer fully open, cube in the drawer). Either way only the command
    tells S2 from S3 or S4 (as in Language-Table and CALVIN, where one start state carries several
    instructions). A cube in the drawer rides in and must stay upright in it."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    in_drawer = bool(rng.random() < S2_CUBE_IN_DRAWER)
    if in_drawer:
        sim.set_drawer(T.DRAWER_OPEN)
        T.place_cube_in_drawer(sim, T.drawer_xy(rng, T.DRAWER_OPEN))
        sim.step(50)
    else:
        sim.set_drawer(float(rng.uniform(*T.DRAWER_PARTLY_OPEN)))
        T.place_cube_on_table(sim, T.table_xy(rng))
    ok = move_drawer(sim, ARM_NAME[arm], 0.0, rec)
    cube_ok = not in_drawer or (T.cube_in_drawer(sim) and sim.data.body("cube").xmat.reshape(3, 3)[2, 2] > 0.95)
    return finish(sim, dict(task="S2", arm=arm, seed=seed, cube_in_drawer=in_drawer, opening=round(sim.drawer_opening(), 3),
                            command=RZ.command("S2", seed, rand, arm=arm),
                            success=ok and sim.drawer_opening() < 0.01 and cube_ok), scene)


S3_DRAWER_OPEN = 0.5  # share of S3 episodes that start with the drawer open, like S2 (see run_S3)


def run_S3(arm: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """S3: move the cube to the middle (S3_TARGET) with `arm`. The cube starts anywhere in the
    table area, drawn exactly as in S1/S2 (T.table_xy keeps it off the target), so the start
    doesn't give the task away. Half the episodes start with the drawer partly open
    (DRAWER_PARTLY_OPEN), exactly like S2's cube-on-table start, so only the command tells S3 from
    S2; the drawer is then a bystander and must not move."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    drawer_open = bool(rng.random() < S3_DRAWER_OPEN)
    if drawer_open:
        sim.set_drawer(float(rng.uniform(*T.DRAWER_PARTLY_OPEN)))
    opening = sim.drawer_opening()
    goal = T.S3_TARGET
    T.place_cube_on_table(sim, T.table_xy(rng))
    a = ARM_NAME[arm]
    picked = pick_cube(sim, a, rec)
    if picked:
        place_cube(sim, a, goal, rec)
    drawer_moved = abs(sim.drawer_opening() - opening)
    return finish(sim, dict(task="S3", arm=arm, seed=seed, drawer_open=drawer_open, picked=picked,
                            drawer_moved=round(drawer_moved, 4),
                            command=RZ.command("S3", seed, rand, arm=arm),
                            success=picked and cube_on_table_at(sim, goal) and drawer_moved < PUSHED), scene)


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
                            success=picked and cube_on_table_at(sim, goal)), scene)


def run_T2(opener: str, target: str = "middle", seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """T2: `opener` opens the drawer (cube inside), the other arm takes the cube out and puts it on
    a named spot (CUBE_TARGETS)."""
    goal = T.CUBE_TARGETS[target]
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    T.place_cube_in_drawer(sim, T.drawer_xy(rng, 0.0))
    sim.step(50)
    picker = "B" if opener == "A" else "A"
    opened = move_drawer(sim, ARM_NAME[opener], T.DRAWER_OPEN, rec)
    sim.step(100, rec)
    picked = opened and pick_cube(sim, ARM_NAME[picker], rec)
    if picked:
        place_cube(sim, ARM_NAME[picker], goal, rec)
    return finish(sim, dict(task="T2", opener=opener, target=target, seed=seed, opened=round(sim.drawer_opening(), 3),
                            picked=picked,
                            command=RZ.command("T2", seed, rand, opener=opener, picker=picker, target=target),
                            success=picked and cube_on_table_at(sim, goal)), scene)


# --- bar: T1 hand-off ------------------------------------------------------------------------
BAR_MEET = np.array([0.0, -0.12, 0.40])  # bar centre during the hand-off
TAKER_TILTS = (40, 50, 30)  # degrees, tried in order; at the standard 20° the taker hits the giver's wrist-camera mount


def holding_body(sim: Sim, arm: str, body: str, half_h: float) -> bool:
    return bool(sim.touching(arm, body)) and sim.body_pos(body)[2] > TABLE_Z + half_h + 0.01


def pick_bar(sim: Sim, arm: str, rec=None, camera_away_from=None) -> bool:
    """Grasp the bar near the arm's own end and lift it to carry height."""
    plan = T.plan_bar_grasp(sim, arm, camera_away_from=camera_away_from)
    if plan is None:
        return False
    approach_item(sim, arm, plan, T.BAR_OPEN, "bar", rec, follow_plan=camera_away_from is not None)
    sim.grip(arm, GRIPPER_CLOSED, 0.5, rec)
    sim.move_line(arm, sim.tip(arm) + 0.03 * T.Z, 0.3, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    with sim.noisy(arm, rec):
        lift = sim.tip(arm).copy()
        lift[2] = max(lift[2], TABLE_Z + CARRY_Z)
        sim.move_line(arm, lift, 0.5, point=plan.point, jaw_perp=plan.jaw_perp, on_step=rec)
    return holding_body(sim, arm, "bar", T.BAR_W / 2)


def bar_lowest(sim: Sim) -> float:
    R, c = sim.data.body("bar").xmat.reshape(3, 3), sim.body_pos("bar")
    half = np.array([T.BAR_L, T.BAR_W, T.BAR_W]) / 2
    return min((c + R @ (half * np.array(sg)))[2] for sg in np.ndindex(2, 2, 2) for sg in [np.array(sg) * 2 - 1])


def place_bar(sim: Sim, arm: str, xy, rec=None):
    """Carry the held bar so its centre is above `xy`, lower it flat onto the table, release, go home."""
    offset = sim.body_pos("bar") - sim.tip(arm)  # bar centre relative to the fingertips; kept while carrying
    point, axis = sim.pointing(arm), sim.data.body("bar").xmat.reshape(3, 3)[:, 0]
    above = np.array([xy[0], xy[1], TABLE_Z + CARRY_Z]) - offset * [1, 1, 0]
    with sim.noisy(arm, rec):
        sim.move_line(arm, above, 1.2, point=point, jaw_perp=axis, waypoints=30, on_step=rec)
    # Correct for drift of the bar in the grip (consistently ~2-3 cm by the end of the carry):
    # read where it actually is and shift the gripper so its centre is above the target.
    error = np.array([xy[0], xy[1], 0.0]) - sim.body_pos("bar") * [1, 1, 0]
    sim.move_line(arm, sim.tip(arm) + error, 0.4, point=point, jaw_perp=axis, on_step=rec)
    drop = sim.tip(arm) - (bar_lowest(sim) - TABLE_Z - 0.005) * T.Z
    sim.move_line(arm, drop, 0.6, point=point, jaw_perp=axis, on_step=rec)
    sim.grip(arm, T.BAR_OPEN, 0.4, rec)
    sim.move_line(arm, sim.tip(arm) - 0.05 * point, 0.4, on_step=rec)
    go_home(sim, arm, rec)


def bar_on_table_at(sim: Sim, xy, tol: float = 0.025) -> bool:
    p = sim.body_pos("bar")
    flat = abs(sim.data.body("bar").xmat.reshape(3, 3)[2, 2]) > 0.95
    on_table = abs(p[2] - (TABLE_Z + T.BAR_W / 2)) < 0.005
    return bool(np.linalg.norm(p[:2] - np.asarray(xy)) < tol and flat and on_table)


def place_bar_start(sim: Sim, rng) -> tuple[float, float]:
    (x0, x1), (y0, y1) = T.BAR_AREA
    xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
    T.place_bar_on_table(sim, xy)
    sim.step(50)
    return xy


def run_T1(giver: str, seed: int = 0, rec=None,
           rand: Randomization = Randomization()) -> dict:
    """T1: `giver` picks the bar by its own end and holds it between the arms; the other arm
    grips the other end (same top grasp), the giver lets go, and the taker puts the bar down
    in the middle. Both arms end at home."""
    sim, scene = new_sim(seed, rand)
    rng = np.random.default_rng(seed)
    place_bar_start(sim, rng)
    g, t = ARM_NAME[giver], ARM_NAME["B" if giver == "A" else "A"]
    result = dict(task="T1", giver=giver, seed=seed, picked=False, transferred=False,
                  command=RZ.command("T1", seed, rand, giver=giver, taker="B" if giver == "A" else "A"))
    # Each wrist-camera mount sticks out behind the gripper along the reach direction; keep both
    # pointing away from the other arm, otherwise the two mounts meet over the bar.
    if not pick_bar(sim, g, rec, camera_away_from=sim.data.body(f"{t}/base").xpos.copy()):
        return finish(sim, result, scene)
    result["picked"] = True
    offset = sim.body_pos("bar") - sim.tip(g)
    point, axis = sim.pointing(g), sim.data.body("bar").xmat.reshape(3, 3)[:, 0]
    with sim.noisy(g, rec):  # bring the bar's centre to the meeting point, same orientation
        sim.move_to(g, BAR_MEET - offset, 1.4, point=point, jaw_perp=axis, on_step=rec)
    sim.step(100, rec)
    giver_base = sim.data.body(f"{g}/base").xpos.copy()
    plan = None
    for tilt in TAKER_TILTS:  # 40° almost always; the others only if the bar is presented slightly rotated
        plan = T.plan_bar_grasp(sim, t, tilt=tilt, camera_away_from=giver_base)
        if plan is not None:
            break
    if plan is None:
        result["reason"] = "no taker grasp"
        return finish(sim, result, scene)
    approach(sim, t, plan, T.BAR_OPEN, rec, follow_plan=True)
    sim.grip(t, GRIPPER_CLOSED, 0.5, rec)
    sim.grip(g, T.BAR_OPEN, 0.4, rec)
    sim.move_line(g, sim.tip(g) - 0.05 * sim.pointing(g), 0.5, on_step=rec)  # giver backs off
    go_home(sim, g, rec)
    result["transferred"] = holding_body(sim, t, "bar", T.BAR_W / 2) and not sim.touching(g, "bar")
    if result["transferred"]:
        place_bar(sim, t, T.BAR_TARGETS["middle"], rec)
    result["success"] = result["transferred"] and bar_on_table_at(sim, T.BAR_TARGETS["middle"])
    return finish(sim, result, scene)

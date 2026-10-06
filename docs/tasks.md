# Tasks — implementation notes

The task list itself is in the [README](../README.md). This page holds the
implementation detail per task: what is in the scene, what is randomised, how
success is checked, and what has and has not been verified. Code:
[`sim/waste_env.py`](../sim/waste_env.py) (`TASKS`), experts in
[`sim/experts.py`](../sim/experts.py).

## Where everything goes

| Waste stream | Items in the scene (YCB scans) | Destination | Destination built? |
|---|---|---|---|
| Plastic recycling | plastic cup, mustard bottle | plastic bin — under-table pull-out drawer | yes |
| Metal recycling | tomato soup can, tuna can, master chef can, potted meat can | metal bin — under-table pull-out drawer | yes |
| Food waste | banana, apple, lemon, peach, pear, orange, plum, strawberry (leftover liquid: not in the scene yet) | food-waste container on the table | not yet |
| Reuse (to wash) | plate, bowl, mug, fork, spoon, knife | "to wash" basket on the table | not yet |

The story's water bottle and drink can are not in the scene: YCB has neither.

The two recycling bins are pull-out drawers under the front edge of the table,
on the arms' side, as in a kitchen: the plastic bin left of centre
(x = −0.10), the metal bin right of centre (x = +0.10). A bin has to be opened
before anything can go in. Closing it again is not part of any task yet.

## Tasks

Each task is one instruction with one success check. One **episode** is one
attempt at one task: `reset(seed)` sets up the table (the seed decides where
the items stand), the arms act, and the episode ends on success or when the
time limit runs out. All tasks share one environment
([`sim/waste_env.py`](../sim/waste_env.py)); a task only sets its instruction,
which items are on the table, where they may stand, its success check and its
time limit.

Arm A is the left arm, arm B the right arm. Only arm A can open the plastic
bin and only arm B can open the metal bin; both arms can drop items into
either open bin.

### T1 `can_to_metal_bin` — implemented

> "Pick up the tomato soup can with arm A, open the metal bin with arm B, and
> put the can in the metal bin with arm A."

| | |
|---|---|
| Steps | 1. Arm A grasps the can and lifts it. 2. Arm B hooks the metal bin's handle and pulls it open, then moves out of the way. 3. Arm A carries the can over the open bin and releases it. |
| Randomised | the can's position (x −0.16 … −0.06, y −0.10 … −0.05) and rotation — nothing else yet (weight, friction, lighting etc. still to add) |
| Success | the can's centre is inside the metal bin |
| Time limit | 25 s |
| Status | scripted expert succeeds on 50/50 seeds (≈ 16 s of simulated time per episode). The expert reads object positions from the simulator and plans the whole episode at reset (open loop). The bin opens to ≈ 0.16 m of its 0.18 m — enough for the drop. |

**Where it is implemented**

| What | Where |
|---|---|
| Task definition (instruction, items, placement, success, time limit) | [`sim/waste_env.py:109`](../sim/waste_env.py#L109) — the `"can_to_metal_bin"` entry in `TASKS` |
| Where the can may stand | `CAN_PATCH`, [`sim/waste_env.py:90`](../sim/waste_env.py#L90) |
| Placing the can at reset (random position + rotation, clear of the arms) | `_place_can`, [`sim/waste_env.py:93`](../sim/waste_env.py#L93) |
| Success check | `_can_in_metal_bin`, [`sim/waste_env.py:104`](../sim/waste_env.py#L104) — the can's centre inside the scene site `metal_bin_interior` |
| Building the scene with the can attached (scene-decisions D17) | `build_model`, [`sim/waste_env.py:48`](../sim/waste_env.py#L48); cached per item set in `_use_items`, [line 142](../sim/waste_env.py#L142) |
| Environment (reset, step, observation, shared by all tasks) | `WasteSortEnv`, [`sim/waste_env.py:119`](../sim/waste_env.py#L119) |
| Scripted expert | `CanToMetalBinExpert`, [`sim/experts.py:74`](../sim/experts.py#L74); the plan is built in `reset()` ([line 120](../sim/experts.py#L120)): arm A grasp and lift ([128](../sim/experts.py#L128)), arm B opens the bin ([160](../sim/experts.py#L160)), arm A drops the can ([178](../sim/experts.py#L178)). Tunables at the top of the file (`GRASP_Z`, `SLIDE`, `PARK_B`, …) |
| Inverse kinematics used by the expert | `ArmIK`, [`sim/ik.py`](../sim/ik.py) |
| Scene parts the task uses | metal bin and its sites `metal_bin_hook` / `metal_bin_interior` in [`sim/scenes/dinner_table.xml`](../sim/scenes/dinner_table.xml); the can model [`sim/assets/objects/ycb/005_tomato_soup_can/`](../sim/assets/objects/ycb/005_tomato_soup_can/) |
| Run the expert | [`scripts/smoke_test.py`](../scripts/smoke_test.py) (success rate, per-phase log, videos in `outputs/smoke/`); [`scripts/view_episode.py`](../scripts/view_episode.py) `--expert` to watch it |
| Record demonstrations (phase 2) | [`scripts/record_states.py`](../scripts/record_states.py) → `data/raw/can_to_metal_bin/seed_<n>.npz`; [`scripts/render_dataset.py`](../scripts/render_dataset.py) → LeRobot dataset in `data/lerobot/can_to_metal_bin/` |
| Tests | [`tests/test_env.py`](../tests/test_env.py): reset places only the can, in its patch ([line 32](../tests/test_env.py#L32)); the expert succeeds on seeds 0, 25, 47 ([line 72](../tests/test_env.py#L72)) |

**Demonstrations recorded (2026-10-06):** seeds 0–49, 50/50 successful, 812
control steps each (16.2 s at 50 Hz, including 0.5 s after success). Rendered
dataset: `data/lerobot/can_to_metal_bin/`, 40,598 frames, cameras `overview`,
`left_wrist`, `right_wrist` at 224×224 (defaults; scene-decisions D26 open),
346 MB. Seeds for evaluation must be kept separate from these (not chosen yet).

### T2 `cup_to_plastic_bin` — planned

> "Pick up the plastic cup with arm B, open the plastic bin with arm A, and put
> the cup in the plastic bin with arm B."

The same structure as T1 with the arms' roles swapped (not an exact mirror:
arm B is rotated, not reflected, so its numbers will differ). Success: the cup
is inside the plastic bin. Checked: arm A can open the plastic bin and arm B
can drop into it. Not checked: arm B's grasp of the cup, and where the cup
may stand.

### T3 `handoff_can_to_metal_bin` — planned

> "Pick up the tomato soup can with arm A, hand it to arm B, and put it in the
> metal bin with arm B."

A hand-off: arm A lets go only once arm B holds the can. Success: the can is
inside the metal bin and was never on the table between the two grasps. Not
checked: whether there is a pose where both arms can hold the can at once.

### T4 `empty_bottle` — planned

> "Hold the food-waste container with arm A, and pour the leftover water from
> the bottle into it with arm B."

Complementary actions: one arm steadies, the other pours. The water is small
spheres (see the scene guide; tested only with a scripted bowl, not with the
arms). Success: a share of the spheres, still to be fixed, ends up in the
container. Needs: the food-waste container and a bottle, both hollow for
collision (YCB has no water bottle), and a bottle grasp by arm B.

### T5 `mug_to_wash_basket` — planned

> "Put the mug in the to-wash basket."

Success: the mug is inside the basket. Needs: the basket, placed where an arm
can reach it — not checked yet, so which arm is open.

### T6 `clear_the_table` — planned

> "Clear the table."

Several items on the table at once. The policy must recognise each item's
waste stream from the cameras and chain the tasks above. Success: every item
is in its correct destination. Builds on T1–T5.

**Coverage of the challenge:** multi-task manipulation (T1–T6), hand-off (T3),
complementary actions (T4), natural-language instructions (every task has one;
T6 requires choosing the steps).


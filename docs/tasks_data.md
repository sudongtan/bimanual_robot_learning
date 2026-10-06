# Tasks and data

How a task is defined and how its demonstration data is made (Part A, the
same for every task), then each task's details (Part B). The task list itself
is in the [README](../README.md); the scene the tasks run in is in
[scene.md](scene.md); training and evaluation are in
[train_evaluation.md](train_evaluation.md). Commands run from the repo root after
`uv sync --extra local` ([pipeline.md](pipeline.md), before you start); scripts
that take a task accept any task in `TASKS` (implemented so far:
`t0_can_to_metal_bin`). References: "Dnn" = a decision in [design-decisions.md](design-decisions.md),
"#nn" = an entry in [debug-log.md](debug-log.md).

---

# Part A — Tasks and data in general

## A1. What a task is

A task is one instruction with one success check, defined as one entry in
`TASKS` in [`sim/waste_env.py`](../sim/waste_env.py). All tasks share one
Gymnasium environment, `WasteSortEnv`; a task only sets:

| Part | What it is |
|---|---|
| `instruction` | the language instruction (also stored in every dataset frame) |
| `items` | which YCB items are on the table; the environment attaches exactly these to the scene on reset (design-decisions D17) |
| `place(env, rng)` | where the items start; random, driven by the episode's seed |
| `success(env)` | checked after every step; the episode ends at the first success |
| `max_seconds` | the time limit (default 25 s); reaching it without success is a failure |

The environment itself: an action is 12 joint targets (6 per arm, radians),
one step is 1/50 s, the observation is the 12 joint positions (`agent_pos`)
plus camera images when asked for (`pixels`), and `info["is_success"]` reports
success — key names chosen to match LeRobot's simulation environments (D28).

## A2. Episodes and seeds

One **episode** is one attempt at a task: `reset(seed)` sets up the table,
the arms act, the episode ends on success or at the time limit. **The seed
decides where the items start** — one seed is one starting scene, and the
same seed always gives the same scene. That is how data and tests are kept
apart (D27):

| Seeds | Used for |
|---|---|
| 0–199 | demonstrations (training data) |
| 1000–1049 | evaluation only — never used for training (train_evaluation.md) |

**Watch an episode** — the scene with the task's items, placed for the seed:

```bash
uv run python scripts/view_episode.py --task t0_can_to_metal_bin --seed 7            # you drive the viewer
uv run mjpython scripts/view_episode.py --task t0_can_to_metal_bin --seed 7 --expert # watch the scripted expert
```

`--expert` needs `mjpython` on macOS (installed with MuJoCo). In the plain
viewer, **Load key** resets the arms and bins but puts items at the world
origin — restart the script to get the episode back. Viewer controls:
scene.md A5.

## A3. The scripted expert

Each task with data has a scripted expert in
[`sim/experts.py`](../sim/experts.py), registered in `EXPERTS`. An expert
reads the true positions of items and bins from the simulator (privileged
information no learned policy gets), plans the whole episode at reset with
inverse kinematics ([`sim/ik.py`](../sim/ik.py)), and plays the plan back —
open loop. It declares which item it grasps and in which phase, so the smoke
test can flag an item that moved before the grasp.

**Smoke-test it:**

```bash
uv run python scripts/smoke_test.py --task t0_can_to_metal_bin                  # seeds 0-4, per-phase log
uv run python scripts/smoke_test.py --task t0_can_to_metal_bin --seeds $(seq 0 49) --quiet
```

Prints per seed: success or failure, each item's start and end position, how
far each bin is open, and a warning if the grasped item moved before the
grasp; saves a video of the first seed (`--video-all` for all) in
`outputs/smoke/<task>/`. Expected for T0: every seed succeeds at ≈ 15.7 s.

## A4. Recording a dataset

In two steps (D20) — physics is cheap (1.8 ms per step), rendering images is
not:

1. **Record states** — [`scripts/record_states.py`](../scripts/record_states.py)
   runs the expert for each seed without rendering and saves
   `data/raw/<task>/seed_<n>.npz`: the full simulator state, the 12 joint
   positions and the 12 actions at every step, the success flag. Recording
   continues 0.5 s after success, then stops (D25).
2. **Render the dataset** — [`scripts/render_dataset.py`](../scripts/render_dataset.py)
   rebuilds each episode's scene, sets every saved state, renders the cameras
   (in parallel processes) and writes a LeRobot v3 dataset to
   `data/lerobot/<task>/`. Only successful episodes go in (D25).

**What one frame holds** (D24, D25): `observation.images.<camera>` for
`overview`, `left_wrist`, `right_wrist` (default 224×224; D26 open),
`observation.state` = the 12 joint positions, `action` = the 12 joint targets
sent at that step, `task` = the instruction. One frame per control step, 50
fps. Images can be re-rendered at another size or with other cameras without
re-running the expert.

**Cost on this Mac** (8 workers, 3 cameras at 224²): ≈ 24 frames/s — ≈ 28 min
for 50 episodes; ≈ 7 MB of states and ≈ 346 MB of dataset per 50 episodes.

**Run:**

```bash
# 1. run the expert, save states: data/raw/<task>/seed_<n>.npz (~0.7 s per episode)
uv run python scripts/record_states.py --task t0_can_to_metal_bin --seeds $(seq 0 199)
# 2. render into a LeRobot dataset: data/lerobot/<task>/ (T0: seeds 0-49, 3 cameras, 224x224)
uv run python scripts/render_dataset.py --task t0_can_to_metal_bin --limit 50 --workers 8
```

`render_dataset.py` options: `--cameras …`, `--size H W`, `--workers N`
(default CPU cores − 2), `--limit N` (first N episodes), `--no-reflections`,
`--include-failures`, `--raw` / `--out`. ⚠ **It deletes its output folder
before writing** (debug-log #31) — for a quick test, write elsewhere:

```bash
uv run python scripts/record_states.py --seeds 100 101 --out outputs/smoke_raw
uv run python scripts/render_dataset.py --raw outputs/smoke_raw --out outputs/smoke_ds --workers 2
```

**Check a dataset:**

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/t0_can_to_metal_bin', root='data/lerobot/t0_can_to_metal_bin')
print(ds.num_episodes, 'episodes,', ds.num_frames, 'frames at', ds.fps, 'fps')
f = ds[0]; print({k: tuple(v.shape) for k, v in f.items() if hasattr(v, 'shape')}); print(f['task'])"
```

Expected for T0: `50 episodes, 40598 frames at 50 fps`, each camera image
`(3, 224, 224)`, state and action `(12,)`, the T0 instruction.

**Messages that look like errors but are not:** a long `torchcodec`
traceback ending in `Falling back to 'pyav'` (no system FFmpeg; LeRobot uses
pyav instead, debug-log #23); `Svt[info]: …` lines (the AV1 video encoder
writing the dataset).

## A5. Adding to a task, and checking it

| To… | Do |
|---|---|
| Put an item on the table | list it in the task's `items` and place it in the task's `place` function (`sim/waste_env.py`); new YCB items: scene.md A5 |
| Add a scripted expert | write the expert class in `sim/experts.py` (with `grasped_item` and `grasp_phase`) and add it to `EXPERTS` |

**Check:** `uv run pytest tests/test_env.py tests/test_task_<task>.py` —
`test_env.py` runs for every task (reset, the expert, success still holding
after the item settles, saved states replaying exactly, the LeRobot
registration); `test_task_<task>.py` holds one task's own checks.

**Outputs:** `outputs/smoke/<task>/` (smoke-test videos), `data/raw/<task>/`
(recorded states), `data/lerobot/<task>/` (datasets) — none in git.

---

# Part B — The tasks

## Where everything goes

| Waste stream | Items in the scene (YCB scans) | Destination | Destination built? |
|---|---|---|---|
| Plastic recycling | plastic cup, mustard bottle | plastic bin — under-table pull-out drawer | yes |
| Metal recycling | tomato soup can, tuna can, master chef can, potted meat can | metal bin — under-table pull-out drawer | yes |
| Food waste | banana, apple, lemon, peach, pear, orange, plum, strawberry (leftover liquid: not in the scene yet) | food-waste container on the table | not yet |
| Reuse (to wash) | plate, bowl, mug, fork, spoon, knife | "to wash" basket on the table | not yet |

The story's water bottle and drink can are not in the scene: YCB has neither.
A bin has to be opened before anything can go in; closing it again is not
part of any task yet. Arm A is the left arm, arm B the right arm. Only arm A
can open the plastic bin and only arm B the metal bin; both arms can drop
items into either open bin (scene.md A2).

## T0 `t0_can_to_metal_bin` — implemented (pipeline smoke test)

The simplest case of T1: the same instruction and success check, but **only
the can on the table**. Used to take the whole pipeline through once before
T1 and the other tasks (design-decisions D30). A policy trained on T0 never
has to pick the can out from other items, and its instruction never changes —
T0 measures the pipeline, not those abilities. Walkthrough: [pipeline.md](pipeline.md).

> "Pick up the tomato soup can with arm A, open the metal bin with arm B, and
> put the can in the metal bin with arm A."

| | |
|---|---|
| On the table | the tomato soup can only |
| Start | the can at a random spot (x −0.16 … −0.06, y −0.10 … −0.05) and rotation — nothing else varies yet (weight, friction, lighting etc. still to add) |
| Success | the can's centre inside the metal bin (the site `metal_bin_interior`, a box that moves with the bin) |
| Time limit | 25 s |

**The expert.** 1. Arm A grasps the can and lifts it. 2. Arm B hooks the
metal bin's handle, pulls it open (to ≈ 0.16 m of its 0.18 m — enough), then
parks beside its own base. 3. Arm A carries the can over the open bin and
releases it. How the grasp works: with the gripper pointing straight down the
fingertips reach only up to z ≈ 0.40, below the can's top (0.417), and no
tilted approach reaches a grasp — so the gripper cannot come down onto the
can. Instead it lowers the open, vertical gripper *beside* the can and slides
it sideways so the can passes between the thin jaws, closes, and lifts with
the wrist free to tilt. The slide-in must start from the side where the wrist
camera housing trails, and the gripper must face the same way before and
during the grasp — otherwise the housing hits the can (debug-log #13, #28).
Other fixes found by tracing failures: arm B parks instead of returning home
(at home its wrist camera was in arm A's drop path, #15); arm A lifts over the
can with the gripper closed and opens only beside it (#16).

**Results.** Expert: 200/200 on seeds 0–199, 50/50 on the evaluation seeds
1000–1049, ≈ 15.7 s per episode.

**Data (2026-10-06).** Recorded states for seeds 0–199 in
`data/raw/t0_can_to_metal_bin/`, all successful (expanded from 50 to cover
the can-patch corner near arm A's base, debug-log #28). **Training dataset:**
`data/lerobot/t0_can_to_metal_bin/`, seeds 0–49 — 50 episodes, 40,598 frames,
346 MB (re-rendered identical after a deletion, debug-log #31). Seeds 50–199
are recorded, not rendered: for a smoke test 50 episodes are enough (D27).

**Where it is implemented**

| What | Where |
|---|---|
| Task definition | the `"t0_can_to_metal_bin"` entry in `TASKS`, [`sim/waste_env.py:113`](../sim/waste_env.py#L113) |
| Where the can may start / placing it | `CAN_PATCH` [line 94](../sim/waste_env.py#L94), `_place_can` [line 97](../sim/waste_env.py#L97) |
| Success check | `_can_in_metal_bin` [line 108](../sim/waste_env.py#L108); site `metal_bin_interior` [`sim/scenes/dinner_table.xml:111`](../sim/scenes/dinner_table.xml#L111) |
| Scripted expert | `CanToMetalBinExpert`, [`sim/experts.py:74`](../sim/experts.py#L74); plan built in `reset()` ([line 135](../sim/experts.py#L135)): arm A grasp and lift ([143](../sim/experts.py#L143)), arm B opens the bin ([181](../sim/experts.py#L181)), arm A drops the can ([199](../sim/experts.py#L199)); tunables at the top (`GRASP_Z`, `SLIDES`, `PARK_B`) |
| Tests | [`tests/test_task_t0_can_to_metal_bin.py`](../tests/test_task_t0_can_to_metal_bin.py) (the can starts in its patch and rests upright; grasp pose reachable at every patch corner; grasp near arm A's base, seeds 1004/1009) and the shared [`tests/test_env.py`](../tests/test_env.py) (reset, expert success, success holds after settling, saved states replay exactly) |

## T1 `can_to_metal_bin` — planned

> "Pick up the tomato soup can with arm A, open the metal bin with arm B, and
> put the can in the metal bin with arm A."

The real version of T0: the same steps and success check, with **other items
on the table** besides the can, so the policy has to find the can among them.
Not built yet. Open: which items and how many, and how they are placed —
including a clearance around the can so the expert's slide-in grasp does not
hit them (design-decisions D30). The code name `can_to_metal_bin` is kept for
it.

## T2 `cup_to_plastic_bin` — planned

> "Pick up the plastic cup with arm B, open the plastic bin with arm A, and put
> the cup in the plastic bin with arm B."

The same structure as T1 with the arms' roles swapped (not an exact mirror:
arm B is rotated, not reflected, so its numbers will differ). Success: the cup
is inside the plastic bin. Checked: arm A can open the plastic bin and arm B
can drop into it. Not checked: arm B's grasp of the cup, and where the cup
may stand.

## T3 `handoff_can_to_metal_bin` — planned

> "Pick up the tomato soup can with arm A, hand it to arm B, and put it in the
> metal bin with arm B."

A hand-off: arm A lets go only once arm B holds the can. Success: the can is
inside the metal bin and was never on the table between the two grasps. Not
checked: whether there is a pose where both arms can hold the can at once.

## T4 `empty_bottle` — planned

> "Hold the food-waste container with arm A, and pour the leftover water from
> the bottle into it with arm B."

Complementary actions: one arm steadies, the other pours. The water is small
spheres (scene.md B7; tested only with a scripted bowl, not with the arms).
Success: a share of the spheres, still to be fixed, ends up in the container.
Needs: the food-waste container and a bottle, both hollow for collision (YCB
has no water bottle), and a bottle grasp by arm B.

## T5 `mug_to_wash_basket` — planned

> "Put the mug in the to-wash basket."

Success: the mug is inside the basket. Needs: the basket, placed where an arm
can reach it — not checked yet, so which arm is open.

## T6 `clear_the_table` — planned

> "Clear the table."

Several items on the table at once. The policy must recognise each item's
waste stream from the cameras and chain the tasks above. Success: every item
is in its correct destination. Builds on T1–T5.

**What the task set exercises:** multi-task manipulation (T1–T6), hand-off
(T3), complementary actions (T4), natural-language instructions (every task
has one; T6 requires choosing the steps) — the abilities the project goal in
the README calls for. T0 is outside this list: it only tests the pipeline.

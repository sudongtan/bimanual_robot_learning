# Bimanual SO-101 workcell: tasks, scene and status

_Last updated: 2026-11-01_

Goal: collect scripted bimanual demos in sim of four primitive skills and two composites built from them, each composite split across the two arms. Use them to (1) train imitation policies (ACT/DP) evaluated in sim, one policy with a one-hot variant input (task × arm × target, 20 variants), checked first with a few per-variant policies, then (2) fine-tune a pretrained VLA (SmolVLA/π0) on all tasks with varied instructions, evaluated on held-out start states and paraphrases. Then test composition: running a composite as its two primitive commands in a row, with and without composite training data.

## Status

The workcell, a scripted expert for all six tasks, the per-episode randomisation, the recorder and the success checks (which score a policy the same way as the expert, `sim/episode.py`) are built and validated. The step-0 batch is recorded and checked ([2data_collection.md](2data_collection.md)). Working to the **minimal plan** (below); results in [3train_evaluation.md](3train_evaluation.md):
- **Step 1 done:** ACT on S1 arm A, 88% (44/50) on eval starts, all successes stopped, with temporal ensembling. Training on `action_nominal` gave 82%, no real difference, so `action` is kept.
- **Steps 2–3 done:** `data/min` (300 episodes); one ACT with the one-hot variant input: S1 72/100, T1 195/200, stop and continue follow the command every time (no S1 rollout took the cube out, every T1 rollout continued, no cube on the wrong side). All S1 failures are the 5 mm slide limit. Held-out colours: S1 77/100, T1 194/200, no gap.
- **Step 4 trained** on a rented RTX 4090 ([4cloud_training.md](4cloud_training.md)), 20k steps; a first check succeeds on 4/5 S1:A rollouts (the failure is the slide). Full evaluation queued on the Mac.
- **Step 4 evaluated** (3train_evaluation.md): SmolVLA is level with ACT on S1, T1 56% against 98% (loose placement; it continues 198/200), no colour gap; under the held-out sentence shape T1 fails 200/200.
- **Step 5 done for ACT and SmolVLA** (3train_evaluation.md): distractors don't matter; the turned cube costs SmolVLA's T1, not ACT's; SmolVLA follows phrasings close to its training ones and fails on other structures and an empty command.
- **Now:** step 3b's evaluation and stress tests (Diffusion Policy, 10 DDIM steps, `scripts/eval_dp.sh`), ends ~2026-11-04 04:30.

Validation, `python -m sim.validate 40` (40 random starts per arm and per target, all randomisation on, expert at 50 Hz as recorded):

| Task | Result |
|---|---|
| S1 open drawer | 80/80 |
| S2 close drawer | 80/80 |
| S3 cube into drawer | 80/80 |
| S4 cube out of drawer | 240/240 |
| T1 open, then take out | 240/240 |
| T2 put in, then close | 80/80 |

All pass with randomisation off too. Colours and lighting are set during validation but don't affect it: nothing is rendered, and the expert reads positions from the simulator. Every episode has no arm-to-arm contact and nothing pushed (distractors, bystanders, and the item being grasped before the jaws close, each < 5 mm), and ends with a still 2 s hold (joints < 0.05 rad, cube and drawer < 5 mm). Joint jumps between IK waypoints are ≤ 0.11 rad (the slow tests allow 0.15).

## Tasks

Arm A = left arm (x < 0), arm B = right arm (x > 0). Every task is done by both arms (or in both directions) from the same start areas, so the command, not the scene, says which arm. All tasks start and end with both arms at home.

| Task | Command (canonical) | Start state | Success |
|---|---|---|---|
| **S1 Open drawer** | "open the drawer with arm {A/B}" | Drawer closed, cube inside | Drawer ≥ 13 cm open; cube kept its place in the drawer (< 5 mm relative to it), upright |
| **S2 Close drawer** | "close the drawer with arm {A/B}" | Half: drawer partly open 6–12 cm, cube on the table; half: drawer fully open 14 cm, cube inside | Drawer < 1 cm open; a cube that started inside is still inside, upright |
| **S3 Cube into drawer** | "put the cube in the drawer with arm {A/B}" | Drawer partly open 6–12 cm, cube on the table | Cube upright on the drawer floor, set down (not dropped); drawer not moved |
| **S4 Cube out of drawer** | "take the cube out of the drawer and put it on the {left/middle/right} of the table with arm {A/B}" | Drawer fully open 14 cm, cube inside | Cube upright within 2 cm of the target |
| **T1 Open, then take out** | "open the drawer with arm {A/B}, then take the cube out and put it on the {left/middle/right} of the table with arm {B/A}" | Drawer closed, cube inside | Cube upright within 2 cm of the target |
| **T2 Put in, then close** | "put the cube in the drawer with arm {A/B}, then close the drawer with arm {B/A}" | Drawer partly open 6–12 cm, cube on the table | Drawer < 1 cm open; cube upright inside |

### Primitives and composites

```
S1 open drawer      ⟷  S2 close drawer         inverse pair
S3 table → drawer   ⟷  S4 drawer → table       inverse pair
T1 = S1 → S4        ⟷  T2 = S3 → S2            inverse composites; the second step is done by the other arm
```

Each composite is two primitive commands, done by different arms. This is LIBERO-Long's structure (long tasks made of two LIBERO-90 primitives; T2 is LIBERO-10's "put the black bowl in the bottom drawer of the cabinet and close it") and CALVIN's (instructions chained in a row), with the hand-over between arms added.

**Stop or continue.** When the first step of a composite is done, the scene is the same as at the end of the primitive: after the drawer is opened (end of S1, middle of T1) and after the cube is put in (end of S3, middle of T2). Only the command says whether to stop or to hand over to the other arm. That only works as a test if the start scene can't give the task away (next section). The opened drawer is also S4's start: a full opening aims past the end stop, so S1 and T1 leave the drawer at 13.9–14 cm, where S2 and S4 start (aiming at the stop itself, the arm lagged and left it at 13.4 cm, a 6 mm cue in the overview). Two things make stopping learnable and measurable: every demonstration ends with a recorded 2 s hold, and rollouts run past the demonstration length (stop and continue rates, in the plan below).

**S3 and S4 vary different things.** S3 varies the source (the cube anywhere on the table) and has one target (the drawer); S4 has one source (the drawer) and varies the target (left/middle/right). Each tests one thing, as LIBERO-Spatial (varied layout) and LIBERO-Goal (varied goals) do. Neither start depends on the target, so the start can't give the target away.

### The scene must not give away the task

If a start scene only ever appears with one task, a policy can learn "this scene means that task" and ignore the command. So start scenes are shared between tasks, as in LIBERO-Goal (all tasks from one scene) and Language-Table (all blocks always on the board). Every task's start is one of three scenes, and every scene has at least two tasks:

| Start scene | Tasks | Episodes at 50 per variant |
|---|---|---|
| Drawer closed, cube inside | S1, T1 | S1 100, T1 300 |
| Drawer partly open (6–12 cm), cube on the table | S2, S3, T2 | S2 50, S3 100, T2 100 |
| Drawer fully open (14 cm), cube inside | S2, S4 | S2 50, S4 300 |

Each scene is drawn by the same code for every task that starts from it (`reset` in `sim/episode.py`; `S2_CUBE_IN_DRAWER = 0.5` splits S2 between the second and third). Item starts never depend on the target.

**Known imbalance:** the counts within a scene are uneven, so always guessing the most common task is right 75% of the time in the first scene (T1) and 86% in the third (S4). The scene doesn't decide the task, but ignoring the command is often rewarded. Balance by sampling within each scene during training (oversample the smaller tasks) if the stop and continue rates show a lean, and check with the swap tests. In the minimal plan's scene (1:2) ACT with the one-hot input showed no lean, so no reweighting was needed (3train_evaluation.md, finding 5).

**How each stage uses this:** the VLA (stage 2) is trained on all tasks and must read the command, so shared scenes matter. A per-variant ACT/DP (stage 1) only ever sees its own variant, so shared scenes don't affect it; a single ACT/DP over several variants can't tell them apart and needs the one-hot variant input (as Meta-World MT10/MT50 adds a task ID). **Rule for new tasks:** start from one of these scenes, or share the start scene with another task, and draw item starts the same way for every variant (no start area that depends on the target).

## Scene (`sim/scenes/workcell.xml`)

- **Frame:** table centre x = y = 0, +x toward arm B, +y toward the cabinet; tabletop at z = 0.315 m; table 0.60 × 0.50 m.
- **Arms:** two SO-101 (MuJoCo Menagerie), facing each other, bases at x = ∓0.20 m. Home: upright over each base, arms 22 cm apart.
- **Drawer:** in a small cabinet on the tabletop at the back centre; pulls 14 cm toward the table centre (end stop). The cabinet top is recessed 5 cm from the front.
- **Items:** cube 4 cm, 30 g; distractors puck, cylinder and pebble (0–2 per episode). Unused items are parked on the floor.
- **Cameras:** `overview` (straight down from 1.25 m, 30° vertical field of view: at 640 × 480 it shows 67 × 50 cm, the table and the distractor zones), `left/wrist_cam`, `right/wrist_cam` (75°). Recorded at 240 × 320 by default (`sim/record.py --size`; 480 × 640 as ACT/ALOHA was too slow to train on this Mac, 2data_collection.md design point 4); the cube covers ~400 px in the overview, ~260 px after resizing to 224 × 224. At home the grippers point up, so the wrist cameras only see the scene once an arm reaches down; the overview is the only view of the start.
- **Simulation:** timestep 2 ms (500 Hz). 12 actuators, position-controlled, in this order: `left/shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, then the same for `right/`.

| Where | Area |
|---|---|
| Cube on the table (start) | x −8…+8 cm, y −20…−15 cm (`TABLE_AREA`) |
| Cube in the drawer (start) | x ±1 cm, y 12.5–15 cm in the drawer's frame (`DRAWER_AREA`) |
| Cube put into the drawer (S3, T2) | the centre of the in-drawer area (`DRAWER_DROP`), so it ends where S4/T1 would pick one up |
| Cube targets left / middle / right (S4, T1) | (−8, −18), (0, −18), (+8, −18) cm (`CUBE_TARGETS`); left = x < 0 = arm A's side, as seen in the overview camera |
| Drawer partly open | 6–12 cm (`DRAWER_PARTLY_OPEN`); above ~13 cm the handle gets in the way of the cube's table area |
| Distractor zones | beside the cabinet x ±15…27, y +10…+21 cm; front corners x ±18…27, y −22…−13 cm; one distractor per zone |

## Randomisation (`sim/randomize.py`)

Every factor is configured in `Randomization`; `Randomization()` is everything on, `NONE` everything off, `split="eval"` uses the held-out colours and phrasings.

| Factor | Kind | Setting |
|---|---|---|
| Item start position | Physical | Uniform over the areas above; drawn in each `run_*` from `default_rng(seed)` |
| Drawer opening (partly open starts) | Physical | Uniform over 6–12 cm |
| Arm start pose | Physical | `start_noise`: σ 0.02 rad on all 5 joints of both arms (as ManiSkill) |
| Demonstration noise | Physical | `demo_noise`: smooth, σ 0.008 rad (~3 mm at the fingertip), free-space moves only (as ACT's `inject_noise`, half size) |
| Cube rotation | Physical | ±45° on the table; ±20° in the drawer (beyond ~25° no grasp in the drawer) |
| Distractors | Physical | 0–2 round objects in the distractor zones; an episode fails if one moves > 5 mm |
| Cube colour | Render-only | 6 train colours (red, green, blue, violet, teal, crimson), 4 held-out eval colours (magenta, cyan, navy, maroon), each chosen to stand out from the arms, table, drawer and cabinet (measured: `2data_collection.md`, "Cube colours"); distractors never share the cube's colour |
| Lighting | Render-only | Main light direction, intensity and tint; headlight level |
| Command wording | Label-only | 3 phrasings per task (`COMMANDS`): the first is canonical, the last is held out for eval |
| Arm bases, cameras, table, mass, friction | — | Fixed |

The factors drawn in `randomize.apply` use their own random stream (`episode_rng`), so switching them on or off doesn't change an episode's item starts or noise. Render-only factors don't affect the expert (it reads simulator state), so they could also be re-applied later by re-rendering saved states.

## How the scripted expert works (`sim/skills.py`, `sim/tasks.py`)

It reads object positions from the simulator, as in ACT, ManiSkill and RoboTwin.

- **Grasps:** from above, tilted 20° along the arm's reach direction. Jaws close across the reach direction on the table, parallel to the drawer front inside the drawer; faces follow the cube's own axes. Cube grasps keep the planned wrist roll and prefer the roll nearer mid-range.
- **Motion:** joint-space moves to a hover point, then straight lines for the precise parts.
- **Putting the cube in the drawer (S3, T2):** S4's in-drawer grasp in reverse. The wrist roll keeps the wrist-camera mount away from the cabinet top: with the drawer barely open, the other roll tucks the mount under the top's front edge, and the way down hits it although the checked start and end poses are clear. The cube is lowered to ~5 mm above the drawer floor (measured) before release; a release higher than 15 mm counts as a failure, because a dropped cube can still land upright by luck.
- **Control rate:** 50 Hz, each command held for 10 physics steps (`CONTROL_EVERY`), so every recorded action is exactly what the actuators got for that frame and replay is exact (`2data_collection.md`, design point 1).
- **Handle grip:** the gripper closes on the drawer handle gradually over 0.4 s; snapping shut drags the drawer ~5 mm at once and slid the cube inside it.
- **Composites:** the first arm returns home before the second arm starts. `sim.subtask` switches from 0 to 1 at that point (the per-frame label of design point 5).
- **Hold:** every episode ends with both arms at home for 2 s (`HOLD_S`), recorded, as the end of ACT's fixed-length scripted episodes. It is one ACT action chunk (100 frames at 50 fps), so a policy is trained on what to do after the task (stay), instead of those steps being padded out (LeRobot's `action_is_pad`). It is the same for every task, so holding still doesn't give away the task. A joint moving > 0.05 rad (`HOLD_STILL`) or the cube or drawer moving > 5 mm during the hold fails the episode.
- **Episode result:** each `run_*` returns a dict with `task`, the command's names (`arm` and `target`, or `opener`/`picker` in T1 and `placer`/`closer` in T2), `seed`, `success`, `checks`, `command`, `scene` (what the randomisation drew), `arm_contacts`, `max_jump`, `pushed`, `distractor_moved`, `hold_motion`, `hold_moved`, `arm_moved` (each arm's largest joint distance from its start), `cube_drop` (the cube's longest fall with no arm touching it), and for S2 the start kind (`cube_in_drawer`). `success` = every entry of `checks` plus the checks on the demonstration itself (`pushed`, the hold).

## Success checks (`sim/episode.py`)

The start (`reset`), the per-step monitor and the success check (`success`) don't use anything from the scripted expert, so a policy rollout is scored the same way as a demonstration (as LIBERO's and ManiSkill's success checks read only the simulator). `success` returns each check by name:

**Why they are separate from the expert.** Each `run_*` used to draw its own start and compute `success` partly from the expert's own steps: whether a grasp plan was found, the cube's height at the moment the expert let go, and `pushed`, which is measured inside the expert's approach. A policy has none of these, so it would have been scored wrongly. A dropped cube and a pushed cube would never be caught, and nothing checked which arm did the work, since the expert uses the named arm by construction. The policy evaluation (`sim/evaluate.py`), the swap and stop tests, and any later reinforcement learning (a reward or a success signal per rollout, as in DPPO or SimpleVLA-RL fine-tuning in sim) all need one check that reads only the simulator, applied to the same starts as the demonstrations. Moving it out of the expert changed nothing for the expert: `sim.validate 40` gives 800/800 before and after, with the same start, command and contact and motion numbers in every episode, and every check passes.

- **The task's condition**, from the Tasks table: drawer open or closed, cube upright in the drawer or within 2 cm of its target, and the extras listed there (S1: cube kept its place in the drawer; S3: drawer didn't move). S3 and T2 also need the cube set down, not dropped: it fell < 15 mm with no arm touching it (`DROPPED`). The expert releases ~5 mm above the floor (6–9 mm measured).
- **Who did it.** In a primitive, the named arm touched its item (drawer or cube), and the other arm never left home (every joint < 0.1 rad from its start, `IDLE_MOVE`, the same threshold as the continue rate). In a composite, each arm touched its own step's item, and the first arm never touched the second step's. Without this, opening the drawer with arm B when told arm A would pass.
- **For every task:** no arm-to-arm contact, no distractor moved > 5 mm.

Two checks apply only to demonstrations, in `skills.finish`: nothing pushed while the expert approaches it (`pushed`, measured inside the expert's approach), and the still 2 s hold (`hold_motion`, `hold_moved`). For policies, the stop rate in the plan below replaces the hold check.

## Key decisions

| Decision | Why |
|---|---|
| Drawer on the tabletop, centred, pulling toward the middle | Under the table edge, the handle was at the edge of reach and the inside hard to see |
| Upright home pose | A diagonal home leaned each arm over the other's working area |
| Commands name the arm | With facing arms 0.40 m apart, a "closer arm" rule puts work right next to a base, where the SO-101 is weakest; it would also let the scene choose the arm |
| 4 cm cube | A 6 cm-tall block tipped over when put down |
| Shared start scenes | So the command, not the scene, decides the task (see above) |
| Primitives in inverse pairs, composites from them | Every primitive is part of a composite, so composition can be tested both ways (T1 and T2) |
| S3 puts the cube in the drawer, not on a table spot | A table target lies in the strip the cube starts from (~16 × 5 cm, between the drawer handle and the arms' reach), so the start gave the target away: with three targets no start area could be ≥ 6 cm from all of them (1.9 cm² left) |
| Success checks read only the simulator (`sim/episode.py`), not the expert | So a policy is scored exactly as a demonstration, by the same check the evaluation, swap tests and any later RL use (see Success checks) |
| The success check includes who did it | The command names the arm, so "the drawer is open" isn't enough: the named arm must do the work and the other stay home. Without it the swap tests couldn't tell a policy that follows the arm in the command from one that uses whichever arm it likes |
| No bar hand-off task | Its scene gave the task away (the bar was only on the table in that task), its start and target areas overlapped the cube's in the same strip, and it isn't built from the primitives. Two arms still work together in T1 and T2 |

## How to run

Use the project's `.venv` Python (`uv run` fails until `policies/lerobot_policy_bc_mlp`, referenced in `pyproject.toml`, exists).

```
python -m pytest tests -m "not slow"    # geometry, reach and randomisation checks, ~40 s
python -m pytest tests                  # plus episodes for every task, arm, target and start kind, ~2.5 min
python -m sim.validate 40               # 40 random starts per arm and per target, all randomisation on, ~5 min
python -m sim.validate 40 none          # everything off
python -m sim.validate 40 cube_yaw      # one factor: start_noise, demo_noise, colours, distractors, cube_yaw, lighting, wording (comma-separated)
python -m sim.validate 40 all eval      # held-out colours and phrasings
```

## Files

| Path | Contents |
|---|---|
| `sim/scenes/workcell.xml` | The scene |
| `sim/assets/so101/` | SO-101 model (MuJoCo Menagerie, Apache-2.0); wrist camera set to 75° |
| `sim/scene.py` | Model loading, IK, controller, demonstration noise, arm-contact monitor |
| `sim/tasks.py` | Task geometry, start areas, grasp planning |
| `sim/episode.py` | What every episode needs, whoever acts: start (`reset`), per-step `Monitor`, `success` |
| `sim/skills.py` | One scripted episode per task: `run_S1` … `run_S4`, `run_T1`, `run_T2` |
| `sim/randomize.py` | `Randomization`: every randomisation factor; command templates |
| `sim/validate.py` | Validation over random starts, per factor |
| `sim/record.py` | Episode recorder: LeRobot dataset plus exact-replay states, see `2data_collection.md` |
| `sim/check_data.py` | First-batch checks on a recorded dataset (images, actions, labels), see `2data_collection.md` |
| `sim/colour_contrast.py` | Cube colour sweep: contrast of each palette (and rejected) colour with its surroundings on a recorded dataset, see `2data_collection.md`, "Cube colours" |
| `sim/evaluate.py` | Policy rollouts in sim under the evaluation conditions in the plan and the step-5 stress conditions (fixed horizon; success, stop rate, continue rate; `--temporal-ensemble` for ACT, `--ddim` for DP); logs each rollout's checks, cube slide and final position |
| `sim/offline_check.py` | Early checks on a checkpoint without the simulator: normalisation floor, replay of training episodes (3train_evaluation.md, finding 10) |
| `sim/residual.py` | Residual RL (minimal plan step 6, optional, parked): base policy plans every `PLAN_EVERY` frames, a bounded residual corrects each frame; scored by the evaluation's `Tracker` |
| `sim/dataset_view.py` | A training view of a recorded dataset: `--variant-input` (adds the one-hot variant vector as `observation.environment_state`), `--nominal` (train on `action_nominal`; used only in step 1) |
| `scripts/` | The runs behind the results: `train_step1.sh` (step 1), `step2_3.sh` (steps 2–3, unattended: record, check, view, train, evaluate), `eval_smolvla.sh` (step 4), `check_dp.sh` (DP checkpoint checks), `stress.sh` (step 5, ACT and SmolVLA), `eval_dp.sh` (step 3b's evaluation and stress tests) |
| `cloud/train.sh`, `4cloud_training.md` | ACT, SmolVLA or DP training on a rented CUDA GPU (LeRobot only; evaluation stays on the Mac), and the step-by-step instructions; `cloud/dp_resume.sh` resumes a run and keeps optimizer state for the latest checkpoint only |
| `tests/` | `test_workcell.py` (geometry, reach), `test_randomize.py` (randomisation, seeds), `test_tasks.py` (episodes, exact replay), `test_episode.py` (starts, success checks), `test_record.py` (recorder, dataset views), `test_evaluate.py` (rollout loop with the expert's commands), `test_residual.py` (the residual-RL episode) |
| `deprecated/` | The earlier design; ignore |

## Minimal plan (active): one scene, S1 and T1

The full plan below records all 20 variants before reaching the VLA. This version reaches it with one start scene, **drawer closed with the cube inside**, shared by S1 and T1, as LIBERO-Goal puts all its tasks in one scene so only the instruction picks the goal. The rest of the full plan is on hold, not dropped: the expert and tests for every task stay as they are.

| Task | Variants | Example command |
|---|---|---|
| S1 open drawer | arm A, arm B | "open the drawer with arm B" |
| T1 open, then take out | opener A or B × target left or right | "open the drawer with arm A, then take the cube out and put it on the right of the table with arm B" |

6 variants, 50 episodes each (300). One scene still shows every point of the project:

| Point | Where it shows |
|---|---|
| The command picks the **arm** | S1 arm A vs arm B, same scene |
| The command picks the **task** | S1 vs T1, same scene |
| The command picks the **target** | T1 left vs right, same scene |
| **Stop or continue** | The end of S1 is the middle of T1 (drawer open, cube inside, arms home): stop rate under S1, continue rate under T1. A policy that ignores the command fails one of the two |
| **Hand-over between arms** | T1: one arm opens, the other takes the cube out |
| **Swap test** | Every variant is evaluated on the same `EVAL_SEEDS`, so the same scene under a different command is already in the results |
| **Language, unseen colours, unseen wording** | The VLA under all three evaluation conditions |

| Step | Record | Train | Check |
|---|---|---|---|
| 1 | S1 arm A: 50 | ACT on S1 arm A, on `action` and on `action_nominal` | As step 1 of the full plan: ≥ 80% on the train condition; keep the better action label. **Done:** 88%; `action` kept |
| 2 | S1 arm B: 50; T1 opener A/B × left/right: 200 (recorded as one dataset with S1 arm A, `data/min`) | — | The first-batch checks (`sim.check_data`). **Done:** 300/300, 11 low-contrast flags (red cubes, dim light) |
| 3 | — | ACT with the variant input (one-hot, as Meta-World MT10/MT50), all 6 variants | Success per variant, swap test, stop rate (S1) and continue rate (T1). **Done:** S1 72%, T1 97.5%; stop/continue and target follow the command |
| 3b | — | Diffusion Policy with the same variant input, all 6 variants (added 2026-11-02) | As step 3: a second imitation-learning baseline on the same scene and input, so step 3's results aren't specific to ACT |
| 4 | — | SmolVLA fine-tuned on the same 300 episodes, with the commands | Same, under the train, eval-colours and eval-phrasing conditions. The gap to step 3 (and 3b) on the train condition is the cost of reading language |
| 5 | — | — | **Stress tests** (replaced residual RL on 2026-11-02, below): steps 3, 3b and 4 under one factor at a time pushed past its training range |
| 6 (optional) | — | Residual RL on a frozen policy, only for a gap step 5 finds | Below |

**Step 5: stress tests (`sim/evaluate.py`, `stress_*` conditions; `scripts/stress.sh`).** Why they replaced residual RL: steps 1–3 leave RL nothing real to fix. Every S1 rollout opens the drawer and the arms hold still; S1's success rate is set by how many slides of 4–6 mm fall under the 5 mm limit (the expert's largest is 4.0 mm, so the limit is our choice, not a task failure), and T1 is at 97.5%. RL on S1 would mostly learn to beat that threshold. Policies near 100% in distribution often fail once one factor moves past the training range: The COLOSSEUM (Pumacay et al. 2024) perturbs RLBench tasks one factor at a time (colours, sizes, camera, lighting, distractors) and finds drops of 30–75%; LIBERO-Plus and LIBERO-PRO (2025) find the same for VLAs at ~95% on LIBERO (object positions, camera, robot start pose, instructions ignored). The question it answers here: **does SmolVLA's pretraining buy robustness that ACT and DP don't have, and under which factor?** Evaluation only, no new data or training.

| Condition | Factor | Expert on the eval starts (S1 A/B, T1 A/B × left/right, 20 each) |
|---|---|---|
| `stress_distractors` | all three distractors every time (training: 0–2) | 120/120 |
| `stress_start_pose` | arm start-pose noise 3× (σ 0.06 rad per joint; training: 0.02) | 120/120 |
| `stress_cube_yaw` | the cube turned up to ±40° in the drawer (training: ±20°) | 118/120 (one start, 100010, T1 opener A: the cube misses its target) |
| `stress_wording` (SmolVLA) | eight more held-out phrasings per task (`RZ.STRESS_COMMANDS`: synonyms, other structures, terse and polite forms), one per rollout; `eval_wording` has only one per task | — (the expert doesn't read commands) |
| `stress_blank_command` (SmolVLA) | an empty command, on S1:A (every variant gets the same input from the same seed). S1 and T1 share the start scene, so stopping or continuing shows whether the command is read at all, as LIBERO-PRO checks with corrupted instructions | — |
| next, with small code changes | the cube's start shifted past the training range; camera pose moved a few cm or degrees; a disturbance mid-episode (cube nudged, drawer pushed back), which tests recovery | — |

Each on S1:A and T1:A:left, 20 rollouts each (the expert passes them, so failures are the policy's); ACT and SmolVLA in `scripts/stress.sh`, DP in `scripts/eval_dp.sh`. Report per policy and condition, against the train condition on the same seeds, with the failed checks.

**Step 6 (optional): residual RL (`sim/residual.py`, written and tested; parked).** Only if step 5 finds a gap a residual can plausibly close, such as recovery from a mid-episode disturbance, the robustness ResiP (Ankile et al. 2024) and Policy Decorator (Yuan et al., ICLR 2025) report. Design: the frozen base plans a chunk from the cameras every `PLAN_EVERY` = 25 frames; a residual adds a bounded correction (≤ `RES_SCALE` = 0.05 rad per joint) from the joints, the base command, the simulator's state and the one-hot variant; PPO on the evaluation's own result, training seeds only. `tests/test_residual.py` checks the wrapper (the expert's commands as base chunks succeed and stop; a zero residual changes nothing). The residual's use of simulator state is privileged information, to be stated with any result.

**Imbalance:** T1 has two thirds of the scene's episodes, so a policy that always continues may be rewarded (though by frames at the decision point, S1's 2 s hold may outweigh it). Recorded 50 per command and trained step 3 without reweighting: no lean (no S1 rollout continued, every T1 rollout did), so none is needed; research and decision in [3train_evaluation.md](3train_evaluation.md), finding 5.

**On hold:** S2, S3, S4 and T2 (their scenes and data); the middle target; the chained-primitive and zero-shot composition tests (steps 13–14, which need the S4 command; the natural next addition is S4 plus S2 from the fully-open scene, so that scene is shared too); π0 and the per-variant T1 baseline (Diffusion Policy moved into the minimal plan as step 3b). What the minimal version gives up: the hand-over is shown inside one command (T1), not as two primitive commands chained.

**Code for the minimal plan (written and checked end to end):**
- Variant input: `RZ.variant_vector` (11 dims, `VARIANT_DIMS`), added to a dataset by `python -m sim.dataset_view SRC DST --variant-input` as `observation.environment_state`, which LeRobot's ACT takes as an extra encoder input, unnormalised. `sim/evaluate.py` supplies it when the policy has it. Checked: ACT trains on it, and a rollout gets the vector.
- Continue rate in `sim/evaluate.py`: once T1's drawer is open (or T2's cube is in), the second arm leaves home before the horizon ends. `test_expert_commands_succeed_in_rollout` replays S1 (stops) and T1 (continues).
- SmolVLA: `bash cloud/train.sh smolvla DATASET OUT` fine-tunes `lerobot/smolvla_base` (LeRobot's recipe: batch 64, 20k steps), mapping our cameras onto its `camera1`–`camera3`. Checked on the Mac (training on MPS, then a rollout through `sim/evaluate.py`). The recipe needs a CUDA GPU: on this Mac's GPU it takes 0.55 s per sample whatever the batch, ~8 days for the 1.28M samples (3train_evaluation.md, finding 6). Steps: [4cloud_training.md](4cloud_training.md).

## Full plan (on hold beyond the minimal version): data and training, step by step

A **variant** is one task with one arm and one target, i.e. one exact command (in T1 the arm is the opener, in T2 the placer). There are 20: S1 2 (arm), S2 2 (arm), S3 2 (arm), S4 6 (arm × target), T1 6 (direction × target), T2 2 (direction).

Every step records new episodes with the recorder (`sim/record.py`, design in `2data_collection.md`), trains, and evaluates in sim before moving on, so each step adds one new thing to check. Counts are per command (design point 6 in `2data_collection.md`): 50 episodes each (as ACT's sim tasks and LIBERO), split evenly over S2's two start kinds. Train only on successful episodes, with balanced sampling within each shared scene (see the known imbalance above). Evaluate 50 rollouts per variant on held-out seeds (`EVAL_SEEDS`, from 100,000, the same for every variant; training uses a unique seed per episode, a block of 500 per variant in 1–10,000, `train_seeds` in `sim/randomize.py`), under each condition separately (as LIBERO-PRO / LIBERO-Plus report each perturbation on its own), so that a change in success can be traced to one cause:

| Condition | Colours | Phrasing | Measures |
|---|---|---|---|
| **Train** | train | train | The task itself. The headline number, and the only one to compare across models |
| **Eval colours** | held-out | train | Generalisation to unseen colours |
| **Eval phrasing** (language-conditioned policies only) | train | held-out | Generalisation to unseen wording |

ACT/DP with the one-hot input never sees the command, so it gets the first two conditions only.

**Horizon and stopping.** Rollouts don't end at success (as LIBERO's do, which would hide whether a policy stops); each runs a fixed horizon, the task's demonstration length (including the 2 s hold) plus 2 s, as CALVIN gives each subtask a fixed horizon. Success is checked at the end. For the primitives that end where a composite continues (S1, S3), two more rates, from the same start scene:

| Rate | Command | Passes if |
|---|---|---|
| **Stop rate** | S1 (or S3) | The task succeeds, and in the last 2 s of the horizon no joint of either arm moves > 0.05 rad and the cube and drawer move < 5 mm (the expert's hold check) |
| **Continue rate** | T1 (or T2) | After the first arm finishes its step, the second arm leaves home (a joint > 0.1 rad from its start) before the horizon ends |

A policy that ignores the command does well on one of the two and badly on the other; one that follows it does well on both. `sim/evaluate.py` runs the rollouts (all three conditions, the stop and continue rates), built on `reset` and `success` in `sim/episode.py`: `python -m sim.evaluate CKPT --variants S1:A --condition train --temporal-ensemble 0.01` (ACT). `test_expert_commands_succeed_in_rollout` drives it with the expert's own commands to check the loop. The second and third conditions are `Randomization(colours_split="eval")` and `Randomization(wording_split="eval")` (`split="eval"` holds out both).

**Variant input (one-hot, 11 dims)**, a separate input for ACT/DP (`observation.environment_state`, which LeRobot's ACT encodes as its own token): task (6: S1, S2, S3, S4, T1, T2) + arm (2: A, B; the opener in T1, the placer in T2) + target (3: left, middle, right; all zero for S1, S2, S3, T2). The recorder saves task, arm and target per episode, so the vector can be built at training time.

**Phase A: check the data with per-variant policies** (plain ACT, no variant input, as in ACT/ALOHA)

| Step | Record | Train | Check |
|---|---|---|---|
| 0 | 5 episodes of each of the 20 variants (100), to test the recorder before the full batch; discarded afterwards, not counted in the 1,000 (the recorder may still change) | — | The first-batch checks in `2data_collection.md` (replay, images, actions, labels) |
| 1 | S1 arm A: 50 | ACT on S1 arm A only | Easiest task. Aim ≥ 80% on the train condition (ACT's scripted transfer cube: 90%, measured without any colour shift); report eval colours alongside. Also compare training on `action` (the command sent, with demonstration noise) vs `action_nominal` (the expert's command without noise) and keep the better one for all later steps. If this fails, fix the data, not the policy |
| 2 | T1 opener A, middle: 50 | ACT on that variant only | Longest task (two arms in turn, ~18 s). Lower is expected (ACT's scripted insertion: 50%); this sets the composite baseline |
| 3 | — | Diffusion Policy on S1 arm A (step 1's data) | DP training and evaluation work |

**Phase B: grow one policy, task by task** (ACT with the variant input; retrain from scratch on everything recorded so far at each step)

| Step | Record | Train on | New thing checked |
|---|---|---|---|
| 4 | S1 arm B: 50 | S1 A + B | The arm input works: same scenes, and the named arm moves. Compare S1 arm A with step 1; it should not drop |
| 5 | S2 arm A, B: 100 | S1, S2 | A second task; S1 should not drop |
| 6 | S3 arm A, B: 100 | S1–S3 | First real task test: drawer partly open + cube on table → S2 or S3. The cube is found wherever it starts |
| 7 | S4 6 variants: 300 | S1–S4 | Drawer fully open + cube inside → S2 or S4. Target input works (cube within 2 cm of the right spot). **Keep this policy:** it is the primitives-only policy for the zero-shot test in step 14 |
| 8 | T1 other 5 variants: 250 (opener A, middle is from step 2) | S1–S4, T1 | Drawer closed + cube inside → S1 or T1. The two arms take turns; the policy stops after S1 when told S1 (stop rate for S1, continue rate for T1) |
| 9 | T2 arm A, B: 100 | All 6 tasks (1,000 episodes) | Drawer partly open + cube on table → S2, S3 or T2. The policy stops after S3 when told S3 (stop rate for S3, continue rate for T2) |

**Phase C: final stage-1 results and the VLA**

| Step | Train on | Check |
|---|---|---|
| 10 | ACT and DP with the variant input, all 1,000 episodes | Success per variant (20 rows). **Swap test:** on each shared start scene, give a different variant input and check the behaviour follows it, not the scene |
| 11 | SmolVLA fine-tuned on all 1,000 episodes, with the commands (train phrasings) | Success per variant under all three conditions. Swap test with a different command (as LIBERO-PRO / LIBERO-Plus). The gap to step 10 **on the train condition** is the cost of reading language instead of a one-hot input. The eval-colours gap is not: it mostly reflects the pretrained vision encoder versus ACT's ResNet, so report it separately |
| 12 | π0, same data (if compute allows) | Same as step 11 |

**Phase D: composition** (evaluation only, no new data)

| Step | Policy | Check |
|---|---|---|
| 13 | Steps 10–11 (trained with T1/T2) | **Chained primitives:** run T1 as the S1 command, then the S4 command when the drawer is open; T2 as S3, then S2 once the cube is in (for ACT/DP, switch the variant input at that point), as CALVIN chains instructions. Compare with the single T1/T2 command |
| 14 | Step 7's policy (S1–S4 only) and a VLA trained on S1–S4 only | **Zero-shot composition:** the same chaining, with no composite data in training |

**Caveat for T2 chaining:** after S3, S2 has to close a partly open drawer with the cube inside. No S2 start looks like that (it would be a start scene only S2 has, which gives the task away). In step 14 that scene was never seen at all; in step 13 it was seen only in the middle of T2, under the T2 command, never under the S2 command. So in both steps T2's chaining loss mixes the switch with a start the S2 command wasn't trained on; report it separately from T1. T1 chaining has no such gap: the end of S1 is S4's start (drawer 13.9–14 cm, cube inside, arms home).

If a step's check fails, stop and fix it there: a problem found in step 6 is much harder to trace once T1 and T2 are added.

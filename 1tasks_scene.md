# Bimanual SO-101 workcell: tasks, scene and status

_Last updated: 2026-10-29_

Goal: collect scripted bimanual demos in sim of four primitive skills and two composites built from them, each composite split across the two arms. Use them to (1) train imitation policies (ACT/DP) evaluated in sim, one policy with a one-hot variant input (task × arm × target, 20 variants), checked first with a few per-variant policies, then (2) fine-tune a pretrained VLA (SmolVLA/π0) on all tasks with varied instructions, evaluated on held-out start states and paraphrases. Then test composition: running a composite as its two primitive commands in a row, with and without composite training data.

## Status

The workcell, a scripted expert for all six tasks, the per-episode randomisation, the recorder and the success checks (which score a policy the same way as the expert, `sim/episode.py`) are built and validated. The step-0 batch is recorded and checked ([2data_collection.md](2data_collection.md)). **Now: plan step 1** (S1 arm A, 50 episodes at 240 × 320, ACT trained on `action` and on `action_nominal`); `sim/evaluate.py` runs the rollouts.

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

**Known imbalance:** the counts within a scene are uneven, so always guessing the most common task is right 75% of the time in the first scene (T1) and 86% in the third (S4). The scene doesn't decide the task, but ignoring the command is often rewarded. Balance by sampling within each scene during training (oversample the smaller tasks), and check with the swap tests.

**How each stage uses this:** the VLA (stage 2) is trained on all tasks and must read the command, so shared scenes matter. A per-variant ACT/DP (stage 1) only ever sees its own variant, so shared scenes don't affect it; a single ACT/DP over several variants can't tell them apart and needs the one-hot variant input (as Meta-World MT10/MT50 adds a task ID). **Rule for new tasks:** start from one of these scenes, or share the start scene with another task, and draw item starts the same way for every variant (no start area that depends on the target).

## Scene (`sim/scenes/workcell.xml`)

- **Frame:** table centre x = y = 0, +x toward arm B, +y toward the cabinet; tabletop at z = 0.315 m; table 0.60 × 0.50 m.
- **Arms:** two SO-101 (MuJoCo Menagerie), facing each other, bases at x = ∓0.20 m. Home: upright over each base, arms 22 cm apart.
- **Drawer:** in a small cabinet on the tabletop at the back centre; pulls 14 cm toward the table centre (end stop). The cabinet top is recessed 5 cm from the front.
- **Items:** cube 4 cm, 30 g; distractors puck, cylinder and pebble (0–2 per episode). Unused items are parked on the floor.
- **Cameras:** `overview` (straight down from 1.25 m, 30° vertical field of view: at 640 × 480 it shows 67 × 50 cm, the table and the distractor zones), `left/wrist_cam`, `right/wrist_cam` (75°). Recorded at 480 × 640 (as ACT/ALOHA); the cube is ~40 px across in the overview, ~14 px after resizing to 224 for π0. At home the grippers point up, so the wrist cameras only see the scene once an arm reaches down; the overview is the only view of the start.
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
| `sim/evaluate.py` | Policy rollouts in sim under the evaluation conditions in the plan (fixed horizon, success and stop rate) |
| `sim/swap_action.py` | A copy of a dataset with `action` and `action_nominal` swapped, to train on the clean labels (plan step 1) |
| `tests/` | `test_workcell.py` (geometry, reach), `test_randomize.py` (randomisation, seeds), `test_tasks.py` (episodes, exact replay), `test_episode.py` (starts, success checks), `test_record.py` (recorder) |
| `deprecated/` | The earlier design; ignore |

## Plan: data and training, step by step

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

A policy that ignores the command does well on one of the two and badly on the other; one that follows it does well on both. `sim/evaluate.py` runs the rollouts (train and eval-colours conditions, and the stop rate; the continue rate is still to add), built on `reset` and `success` in `sim/episode.py`: `python -m sim.evaluate CKPT --variants S1:A --condition train`. `test_expert_commands_succeed_in_rollout` drives it with the expert's own commands to check the loop. The second and third conditions are `Randomization(colours_split="eval")` and `Randomization(wording_split="eval")` (`split="eval"` holds out both).

**Variant input (one-hot, 11 dims)**, appended to `observation.state` for ACT/DP: task (6: S1, S2, S3, S4, T1, T2) + arm (2: A, B; the opener in T1, the placer in T2) + target (3: left, middle, right; all zero for S1, S2, S3, T2). The recorder saves task, arm and target per episode, so the vector can be built at training time.

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

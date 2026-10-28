# Bimanual SO-101 workcell: tasks, scene and status

_Last updated: 2026-10-28_

Goal: collect scripted bimanual demos in sim to (1) train imitation policies (ACT/DP) evaluated in sim, one policy with a one-hot variant input (task × arm × target, 20 variants), checked first with a few per-variant policies, then (2) fine-tune a pretrained VLA (SmolVLA/π0) on all tasks with varied instructions, evaluated on held-out start states and paraphrases.

## Status

The workcell, a scripted expert for all six tasks and the per-episode randomisation are built and validated. **Next step: recording data** (see [2data_collection.md](2data_collection.md)).

Validation, `python -m sim.validate 40` (40 random starts per arm and per target, all randomisation on):

| Task | Result |
|---|---|
| S1 open drawer | 80/80 |
| S2 close drawer | 80/80 |
| S3 move cube | 80/80 |
| S4 cube out of drawer | 240/240 |
| T1 bar hand-off | 80/80 |
| T2 open, then take out | 240/240 |

All pass with randomisation off too. Colours and lighting are set during validation but don't affect it: nothing is rendered, and the expert reads positions from the simulator. Every episode has no arm-to-arm contact and nothing pushed (distractors, bystanders, and the item being grasped before the jaws close, each < 5 mm). Joint jumps between IK waypoints are ≤ 0.11 rad (the slow tests allow 0.15).

## Tasks

Arm A = left arm (x < 0), arm B = right arm (x > 0). Every task is done by both arms (or in both directions) from the same start areas, so the command, not the scene, says which arm. All tasks start and end with both arms at home.

| Task | Command (canonical) | Start state | Success |
|---|---|---|---|
| **S1 Open drawer** | "open the drawer with arm {A/B}" | Drawer closed. Half: cube on the table; half: cube inside the drawer | Drawer ≥ 13 cm open; cube not moved (< 5 mm; if inside the drawer, relative to the drawer, and upright) |
| **S2 Close drawer** | "close the drawer with arm {A/B}" | Half: drawer partly open 6–12 cm, cube on the table; half: drawer fully open 14 cm, cube inside | Drawer < 1 cm open; a cube that started inside is still inside, upright |
| **S3 Move cube** | "move the cube to the middle with arm {A/B}" | Cube on the table (the same start area as S1/S2, see below). Half: drawer closed; half: drawer partly open 6–12 cm | Cube upright within 2 cm of the middle spot; drawer not moved |
| **S4 Cube out of drawer** | "take the cube out of the drawer and put it on the {left/middle/right} of the table with arm {A/B}" | Drawer fully open 14 cm, cube inside | Cube upright within 2 cm of the target |
| **T1 Bar hand-off** | "pick up the bar with arm {A/B}, hand it to arm {B/A} and put it down on the middle of the table" | Drawer closed, bar on the table (cube parked off-table) | Taker held the bar after the giver let go; bar flat within 2.5 cm of the middle spot |
| **T2 Open, then take out** | "open the drawer with arm {A/B}, then take the cube out and put it on the {left/middle/right} of the table with arm {B/A}" | Drawer closed, cube inside | Cube upright within 2 cm of the target |

T2 = the S1 command, then the S4 command with the same target, by the other arm.

**S3 and S4 split the two skills:** S3 varies the source (anywhere on the table) and keeps one target; S4 keeps one source (the drawer) and varies the target. Each tests one thing, as LIBERO-Spatial (varied layout) and LIBERO-Goal (varied goal) do. A start that depended on the target would give the target away: S3 used to have all three targets and redrew the cube until it was ≥ 6 cm from its own target, so a cube at x = −4 cm only ever went right. S4 and T2 can keep three targets, because their cube starts in the drawer the same way for every target.

### The scene must not give away the task

If a start scene only ever appears with one task, a policy can learn "this scene means that task" and ignore the command. So start scenes are shared between tasks, as in LIBERO-Goal (all tasks from one scene) and Language-Table (all blocks always on the board):

| Start scene | Tasks |
|---|---|
| Drawer closed, cube on the table | S1, S3 |
| Drawer closed, cube inside | S1, T2 |
| Drawer partly open (6–12 cm), cube on the table | S2, S3 |
| Drawer fully open (14 cm), cube inside | S2, S4 |

Implemented in `run_S1`, `run_S2`, `run_S3` (`S1_CUBE_IN_DRAWER`, `S2_CUBE_IN_DRAWER`, `S3_DRAWER_OPEN`, each 0.5). **Known exception:** only in T1 is the bar on the table and the cube off it, so the scene gives away "T1" (not which arm gives it). Also, the bar's start area is within ~3 cm of its target, so T1's success mostly measures the transfer, not the put-down. Fixing both needs a bar area clear of the cube areas and targets; revisit if the training pilot shows the policy ignoring commands.

**How each stage uses this:** the VLA (stage 2) is trained on all tasks and must read the command, so shared scenes matter. A per-variant ACT/DP (stage 1) only ever sees its own variant, so shared scenes don't affect it; a single ACT/DP over several variants can't tell them apart and needs the one-hot variant input (as Meta-World MT10/MT50 adds a task ID). **Rule for new tasks:** start from one of these scenes, or share the start scene with another task, and draw item starts the same way for every variant (no start area that depends on the target).

## Scene (`sim/scenes/workcell.xml`)

- **Frame:** table centre x = y = 0, +x toward arm B, +y toward the cabinet; tabletop at z = 0.315 m; table 0.60 × 0.50 m.
- **Arms:** two SO-101 (MuJoCo Menagerie), facing each other, bases at x = ∓0.20 m. Home: upright over each base, arms 22 cm apart.
- **Drawer:** in a small cabinet on the tabletop at the back centre; pulls 14 cm toward the table centre (end stop).
- **Items:** cube 4 cm, 30 g; bar 12 × 3 × 3 cm, 30 g (T1 only); distractors puck, cylinder and pebble (0–2 per episode). Unused items are parked on the floor.
- **Cameras:** `overview` (straight down from 1.25 m, 45° field of view), `left/wrist_cam`, `right/wrist_cam` (75°). Training images 224 × 224; the cube is ~12 px across in the overview.
- **Simulation:** timestep 2 ms (500 Hz). 12 actuators, position-controlled, in this order: `left/shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, then the same for `right/`.

| Where | Area |
|---|---|
| Cube on the table | x −8…+8 cm, y −20…−15 cm, at least 3 cm from the middle target (`START_CLEARANCE`); the same for S1, S2 and S3 (`table_xy`) |
| Cube in the drawer | x ±1 cm, y 12.5–15 cm in the drawer's frame |
| Bar on the table | centre x ±2 cm, y −20…−15 cm, lying along x |
| Cube targets left / middle / right | (−8, −18), (0, −18), (+8, −18) cm (`CUBE_TARGETS`); S3 uses the middle only (`S3_TARGET`), S4 and T2 all three; left = x < 0 = arm A's side, as seen in the overview camera |
| Bar target | middle only, (0, −17.5) cm |
| Drawer partly open | 6–12 cm (`DRAWER_PARTLY_OPEN`); above ~13 cm the handle gets in the way of the cube's table area |
| Distractor zones | beside the cabinet x ±15…27, y +10…+21 cm; front corners x ±18…27, y −22…−13 cm; one distractor per zone |

## Randomisation (`sim/randomize.py`)

Every factor is configured in `Randomization`; `Randomization()` is everything on, `NONE` everything off, `split="eval"` uses the held-out colours and phrasings.

| Factor | Kind | Setting |
|---|---|---|
| Item start position | Physical | Uniform over the areas above; drawn in each `run_*` from `default_rng(seed)` |
| Arm start pose | Physical | `start_noise`: σ 0.02 rad on all 5 joints of both arms (as ManiSkill) |
| Demonstration noise | Physical | `demo_noise`: smooth, σ 0.008 rad (~3 mm at the fingertip), free-space moves only (as ACT's `inject_noise`, half size) |
| Cube rotation | Physical | ±45° on the table; ±20° in the drawer (beyond ~25° no grasp in the drawer) |
| Distractors | Physical | 0–2 round objects in the distractor zones; an episode fails if one moves > 5 mm |
| Item colour | Render-only | 6 train colours, 4 held-out eval colours; cube and bar never the same colour; distractors never share the items' colours |
| Lighting | Render-only | Main light direction, intensity and tint; headlight level |
| Command wording | Label-only | 3 phrasings per task (`COMMANDS`): the first is canonical, the last is held out for eval |
| Arm bases, cameras, table, mass, friction | — | Fixed |

The factors drawn in `randomize.apply` use their own random stream (`episode_rng`), so switching them on or off doesn't change an episode's item starts or noise. Render-only factors don't affect the expert (it reads simulator state), so they could also be re-applied later by re-rendering saved states.

## How the scripted expert works (`sim/skills.py`, `sim/tasks.py`)

It reads object positions from the simulator, as in ACT, ManiSkill and RoboTwin.

- **Grasps:** from above, tilted 20° along the arm's reach direction. Jaws close across the reach direction on the table, parallel to the drawer front inside the drawer; faces follow the cube's own axes. Cube grasps keep the planned wrist roll and prefer the roll nearer mid-range.
- **Motion:** joint-space moves to a hover point, then straight lines for the precise parts.
- **T1:** each arm grips the bar ~3 cm from its own end; the taker approaches at 40° (fallback 50° / 30°); wrist-camera mounts are rolled away from the other arm; the bar's actual position is corrected for before putting it down.
- **T2:** the opener returns home before the other arm reaches into the drawer.
- **Episode result:** each `run_*` returns a dict with `task`, the arm/target arguments, `seed`, `success`, `command`, `scene` (what the randomisation drew), `arm_contacts`, `max_jump`, `pushed`, `distractor_moved`, and for S1–S3 the start kind (`cube_in_drawer` / `drawer_open`).

## Key decisions

| Decision | Why |
|---|---|
| Drawer on the tabletop, centred, pulling toward the middle | Under the table edge, the handle was at the edge of reach and the inside hard to see |
| Upright home pose | A diagonal home leaned each arm over the other's working area |
| Commands name the arm | With facing arms 0.40 m apart, a "closer arm" rule puts work right next to a base, where the SO-101 is weakest |
| 4 cm cube | A 6 cm-tall block tipped over when put down |
| Bar, not cube, for the hand-off | On a 4 cm cube the two grippers' jaws always collide (0 of 6,480 setups clear, `sim/handoff.py`) |
| Shared start scenes | So the command, not the scene, decides the task (see above) |
| S3 to the middle only; targets only in S4/T2 | With three S3 targets in the cube's start strip, no start area could be ≥ 6 cm from all of them (1.9 cm² left), so the start gave the target away |



## How to run

Use the project's `.venv` Python (`uv run` fails until `policies/lerobot_policy_bc_mlp`, referenced in `pyproject.toml`, exists).

```
python -m pytest tests -m "not slow"    # geometry, reach and randomisation checks, ~50 s
python -m pytest tests                  # plus episodes for every task, arm, target and start kind, ~3 min
python -m sim.validate 40               # 40 random starts per arm and per target, all randomisation on, ~4 min
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
| `sim/skills.py` | One scripted episode per task: `run_S1` … `run_S4`, `run_T1`, `run_T2` |
| `sim/randomize.py` | `Randomization`: every randomisation factor; command templates |
| `sim/validate.py` | Validation over random starts, per factor |
| `sim/handoff.py` | Record of why the cube can't be handed over |
| `sim/record.py` (to write) | Episode recorder, see `2data_collection.md` |
| `sim/evaluate.py` (to write) | Policy rollouts in sim under the evaluation conditions in the plan |
| `tests/` | `test_workcell.py` (geometry, reach), `test_randomize.py` (randomisation), `test_tasks.py` (episodes) |
| `deprecated/` | The earlier design; ignore |

## Plan: data and training, step by step

A **variant** is one task with one arm and one target, i.e. one exact command (in T1 the arm is the giver, in T2 the opener). There are 20: S1 2 (arm), S2 2 (arm), S3 2 (arm), S4 6 (arm × target), T1 2 (direction), T2 6.

Every step records new episodes with the recorder (`sim/record.py`, design in `2data_collection.md`), trains, and evaluates in sim before moving on, so each step adds one new thing to check. Counts are per command (design point 6 in `2data_collection.md`): 50 episodes each (as ACT's sim tasks and LIBERO), split evenly over the start kinds of S1–S3. Train only on successful episodes. Evaluate 50 rollouts per variant on held-out seeds, under each condition separately (as LIBERO-PRO / LIBERO-Plus report each perturbation on its own), so that a change in success can be traced to one cause:

| Condition | Colours | Phrasing | Measures |
|---|---|---|---|
| **Train** | train | train | The task itself. The headline number, and the only one to compare across models |
| **Eval colours** | held-out | train | Generalisation to unseen colours |
| **Eval phrasing** (language-conditioned policies only) | train | held-out | Generalisation to unseen wording |

ACT/DP with the one-hot input never sees the command, so it gets the first two conditions only. There is no policy evaluation script yet (`sim/evaluate.py`, to write). When it is written: `split` in `Randomization` currently switches colours and phrasings together, so it has to become one setting per factor to run the second and third conditions.

**Variant input (one-hot, 11 dims)**, appended to `observation.state` for ACT/DP: task (6: S1, S2, S3, S4, T1, T2) + arm (2: A, B; the giver in T1, the opener in T2) + target (3: left, middle, right; all zero for S1, S2, S3, T1). The recorder saves task, arm and target per episode, so the vector can be built at training time.

**Phase A: check the data with per-variant policies** (plain ACT, no variant input, as in ACT/ALOHA)

| Step | Record | Train | Check |
|---|---|---|---|
| 0 | 5 episodes of each of the 20 variants (100), to test the recorder before the full batch; discarded afterwards, not counted in the 1,000 (the recorder may still change) | — | The first-batch checks in `2data_collection.md` (replay, images, actions, labels) |
| 1 | S1 arm A: 50 | ACT on S1 arm A only | Easiest task. Aim ≥ 80% on the train condition (ACT's scripted transfer cube: 90%, measured without any colour shift); report eval colours alongside. Also compare training on `action` (the command sent, with demonstration noise) vs `action_nominal` (the expert's command without noise) and keep the better one for all later steps. If this fails, fix the data, not the policy |
| 2 | T1 giver A: 50 | ACT on T1 giver A only | Hardest task. Lower is expected (ACT's scripted insertion: 50%); this sets the T1 baseline |
| 3 | — | Diffusion Policy on S1 arm A (step 1's data) | DP training and evaluation work |

**Phase B: grow one policy, task by task** (ACT with the variant input; retrain from scratch on everything recorded so far at each step)

| Step | Record | Train on | New thing checked |
|---|---|---|---|
| 4 | S1 arm B: 50 | S1 A + B | The arm input works: same scenes, and the named arm moves. Compare S1 arm A with step 1; it should not drop |
| 5 | S2 arm A, B: 100 | S1, S2 | A second task; S1 should not drop |
| 6 | S3 arm A, B: 100 | S1–S3 | First real task test: drawer closed + cube on table → S1 or S3; drawer partly open + cube on table → S2 or S3. The cube is found wherever it starts |
| 7 | S4 6 variants: 300 | S1–S4 | Drawer fully open + cube inside → S2 or S4. Target input works (cube within 2 cm of the right spot) |
| 8 | T1 giver B: 50 (giver A is from step 2) | S1–S4, T1 | T1 inside the multi-task policy vs step 2 |
| 9 | T2 6 variants: 300 | All 6 tasks (1,000 episodes) | Drawer closed + cube inside → S1 or T2; the two arms take turns. Also check the policy stops after S1 when told S1 |

**Phase C: final stage-1 results and the VLA**

| Step | Train on | Check |
|---|---|---|
| 10 | ACT and DP with the variant input, all 1,000 episodes | Success per variant (20 rows). **Swap test:** on shared start scenes, give a different variant input and check the behaviour follows it, not the scene (T1's scene is not shared, so leave it out) |
| 11 | SmolVLA fine-tuned on all 1,000 episodes, with the commands (train phrasings) | Success per variant under all three conditions. Swap test with a different command (as LIBERO-PRO / LIBERO-Plus), again leaving out T1's scene. The gap to step 10 **on the train condition** is the cost of reading language instead of a one-hot input. The eval-colours gap is not: it mostly reflects the pretrained vision encoder versus ACT's ResNet, so report it separately |
| 12 | π0, same data (if compute allows) | Same as step 11 |

**Probe after step 11 (evaluation only, no training data):** "move the cube to the left/right" from table starts. It checks whether the policy combines S3's source (a table cube) with S4's target words, or has tied "left/right" to the drawer motion. Not a training task: its starts would never be in the middle, which would give it away.

If a step's check fails, stop and fix it there: a problem found in step 6 is much harder to trace once T1 and T2 are added.

# Bimanual SO-101: does the policy follow the command, or the scene?

A simulated two-arm SO-101 workcell (MuJoCo) for testing whether imitation-learning policies actually follow the instruction they're given. The tasks are designed so that **the start scene never tells the policy what to do**: the same scene appears with several tasks, either arm can be asked to act, and where objects start never depends on the goal.

*Status: the workcell, scripted expert and randomisation are built and validated. Data recording and training are next; no policy results yet.*

## 1. The question

Language-conditioned robot policies are usually judged by their success rate. But success doesn't show *why* a policy succeeded. If a start scene only ever appears with one task, a policy can learn "this scene means that task" and ignore the instruction entirely. LIBERO-PRO and LIBERO-Plus found exactly this in policies that score well on LIBERO: many still succeed when the instruction is corrupted or removed.

This project asks:

> **When the scene can't give the task away, how well do policies follow the command, and what does reading language cost compared with being told the task directly?**

To answer it, the same demonstrations train two kinds of policy:

- **ACT and Diffusion Policy with a one-hot variant input** (task, arm, target), the way Meta-World MT10/MT50 adds a task ID. The policy is told exactly what to do.
- **A fine-tuned vision-language-action model** (SmolVLA, then π0 if compute allows), which has to read the command in natural language.

Both are evaluated in the same scenes, with each kind of change (unseen colours, unseen wording) measured on its own, plus **swap tests**: keep the scene, change the command, and check that the behaviour follows the command.

## 2. Design

### The workcell

Two SO-101 arms (MuJoCo Menagerie model) face each other across a 60 × 50 cm table, with a one-drawer cabinet at the back centre. Items: a 4 cm cube, a 12 cm bar (for the hand-off), and 0–2 distractors. Cameras: one overhead, one on each wrist.

### Tasks

Arm A is the left arm, arm B the right. **The command names the arm**, and every task can be done by either arm from the same start scenes, so the scene never decides which arm moves. (In ALOHA and RoboTwin, the arm is usually fixed by the task or by which side the object is on.)

| Task | Command | Varies |
|---|---|---|
| S1 Open drawer | "open the drawer with arm A" | arm |
| S2 Close drawer | "close the drawer with arm B" | arm |
| S3 Move cube | "move the cube to the middle with arm A" | arm, cube start anywhere on the table |
| S4 Cube out of drawer | "take the cube out of the drawer and put it on the left of the table with arm B" | arm, target (left/middle/right) |
| T1 Bar hand-off | "pick up the bar with arm A, hand it to arm B and put it down on the middle of the table" | direction |
| T2 Open, then take out | "open the drawer with arm A, then take the cube out and put it on the right of the table with arm B" | direction, target |

That gives 20 variants (exact commands), with 50 scripted demonstrations each.

### Shared start scenes

Each start scene appears with more than one task, as in LIBERO-Goal (every task from one scene) and Language-Table (all blocks always on the board):

| Start scene | Tasks that start from it |
|---|---|
| Drawer closed, cube on the table | S1, S3 |
| Drawer closed, cube in the drawer | S1, T2 |
| Drawer partly open, cube on the table | S2, S3 |
| Drawer fully open, cube in the drawer | S2, S4 |

Seeing the start scene leaves at least two possible tasks, so the policy has to use the command.

### Auditing for leaks

Sharing scenes is not enough: **how** objects are placed can leak information too. Before recording, every start-state factor was checked for any dependence on the task, the arm or the target. These include item positions, drawer opening, cube rotation, distractors, colours, lighting, and where unused items are parked.

One example shaped the final task design. S3 originally had three targets (left, middle, right) and redrew the cube's start until it was at least 6 cm from *its own* target. That looks harmless, but it meant each target had its own start region:

| Target | Where the cube could start |
|---|---|
| left | x ≥ −2.8 cm (52 cm²) |
| middle | only the two outer edges, \|x\| ≥ 5.2 cm (23 cm²) |
| right | x ≤ +2.8 cm (52 cm²) |

A cube at x = −4 cm only ever went right, so the policy could learn "cube here → go right" without reading the target word. Keeping every start 6 cm from *all three* targets would have left 1.9 cm² of usable table, because the targets lie inside the strip both arms can reach (about 16 × 5 cm, limited by the drawer handle's sweep at the back and by the arms' reach at the front).

The fix was to give each task one thing to vary:

- **S3 varies the source:** the cube starts anywhere on the table and always goes to the middle.
- **S4 varies the target:** the cube always starts in the drawer and goes to the left, middle or right.

This mirrors the split between LIBERO-Spatial (varied layout) and LIBERO-Goal (varied goals). S1, S2 and S3 now draw the table cube from one shared region: the table area minus a 3 cm disc around the middle target, about 55 of 80 cm². The 3 cm sits above the 2 cm success radius, so no start counts as already solved. Whether the policy can *combine* the two skills (a table cube sent to the left) is kept as an evaluation-only probe, with no training data.

Two leaks remain and are measured, not hidden:

- **T1's scene gives the task away:** the bar is only on the table in T1.
- **Task counts within shared scenes are uneven:** for example, the "drawer closed, cube inside" scene has 300 T2 episodes and 50 S1 episodes, so always guessing T2 there would be right 86% of the time. Swap tests and balanced sampling during training address this.

### Scripted expert and randomisation

A scripted expert reads object poses from the simulator (as in ACT, ManiSkill and RoboTwin) and runs every task with either arm. It is validated at 100% success over 40 random starts per arm and target, with all randomisation on and with it off. No episode has arm-to-arm contact or pushes anything by more than 5 mm.

Each episode randomises:

- **Physical:** item positions, arm start pose, cube rotation, distractors, and smooth demonstration noise in the style of ACT's `inject_noise`.
- **Render-only:** colours (six for training, four held out) and lighting.
- **Label-only:** command wording (two phrasings for training, one held out).

The full scene, task and decision record is in [1tasks_scene.md](1tasks_scene.md). The recording design is in [2data_collection.md](2data_collection.md).

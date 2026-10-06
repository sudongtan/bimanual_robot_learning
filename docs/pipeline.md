# The pipeline, walked through with T0

T0 (`t0_can_to_metal_bin`) is T1 with only the tomato soup can on the table:
arm A picks up the can, arm B opens the metal bin, arm A drops the can in. It
is the smoke test that takes the pipeline through once before the real tasks
(D30). This page follows T0 phase by phase; the details are
in [scene.md](scene.md), [tasks_data.md](tasks_data.md) and
[train_evaluation.md](train_evaluation.md). Phases follow the README; 6 and 7
are optional and left out, 8 (deployment) is pending. References: "Dnn" = a
decision in [design-decisions.md](design-decisions.md), "#nn" = an entry in
[debug-log.md](debug-log.md).

**The story:** build a world with a task in it (1) → a scripted expert solves
the task, giving the demonstrations to learn from and the yardstick to measure
with (2) → three learners, each adding one idea to the last — plain imitation
(3), predicting chunks of actions (4), language and pretraining (5) — trained
on the same demonstrations and measured with the same yardstick → deploy (8,
pending).

**Scoreboard** — T0, evaluation seeds 1000–1049 (50 episodes, never trained on):

| Policy | Phase | Success |
|---|---|---|
| Scripted expert | 2 | **50/50 (100 %)**, ≈ 15.7 s per episode |
| BC (`bc_mlp`) | 3 | trained, not yet evaluated |
| ACT | 4 | training |
| SmolVLA | 5 | not started |

## Before you start

```bash
uv sync --extra local     # Python 3.12, MuJoCo, LeRobot (dataset, training, SmolVLA extras), our BC plugin, OpenVINO
uv run pytest             # all 68 tests, ~35 s; -m "not slow" skips the IK searches and expert runs
```

Commands run from the repo root. Each document holds the commands for its
part: the scene and viewer in [scene.md](scene.md) A5; episodes, the expert
and datasets in [tasks_data.md](tasks_data.md) Part A; training, evaluation
and deployment in [train_evaluation.md](train_evaluation.md). Results land in
`data/` and `outputs/`, not in git.

---

## 1. The world and the task

The scene: a table, two SO-101 arms, two pull-out bins, three cameras
([scene.md](scene.md)). T0 is one entry in `TASKS` in `sim/waste_env.py`: the
can starts at a random spot set by the episode's seed; success = the can's
centre inside the metal bin; 25 s limit ([tasks_data.md](tasks_data.md) A1,
T0).

```bash
uv run python scripts/view_episode.py --task t0_can_to_metal_bin --seed 7   # look at one starting scene
```

**Result:** the scene runs at 10.8× real time; the scene and task tests pass
(`uv run pytest`, before you start).

## 2. The expert: demonstrations, and preparing evaluation

**Demonstrations.** The scripted expert grasps the can from the side, arm B
pulls the bin open, the can is dropped in ([tasks_data.md](tasks_data.md) T0).
Its episodes are recorded as states, then rendered into a LeRobot dataset
(tasks_data.md A4).

```bash
uv run python scripts/smoke_test.py --task t0_can_to_metal_bin                                # quick check, video
uv run python scripts/record_states.py --task t0_can_to_metal_bin --seeds $(seq 0 199)       # ~2 min
uv run python scripts/render_dataset.py --task t0_can_to_metal_bin --limit 50 --workers 8    # ~28 min
```

**Result:** states for seeds 0–199, all successful; the training dataset
`data/lerobot/t0_can_to_metal_bin/` holds seeds 0–49: 50 episodes, 40,598
frames, 3 cameras at 224×224.

**Preparing evaluation.** Before anything is learned, the yardstick is fixed
so every policy is measured on the same test: the evaluation seeds 1000–1049,
the task's own success check, LeRobot's evaluator connected to our
environment, an evaluator for the expert, the expert scored, and the chain
tested with a trained checkpoint ([train_evaluation.md](train_evaluation.md)
§3a — six steps).

```bash
scripts/evaluate.sh t0_can_to_metal_bin expert                # ~80 s
```

**Result:** the expert scores **50/50** — after fixing a grasp bug that the
first run (48/50) exposed (#28). It is the ceiling the learners aim for: they
only ever see what the expert did.

## 3. BC — plain imitation

The simplest learner: our network `bc_mlp` looks at the current camera images
and joint positions and predicts the next action, one step at a time
(train_evaluation.md §1). Its score shows how far plain imitation of the
expert gets:

```bash
scripts/train_policy.sh t0_can_to_metal_bin bc_mlp            # 20k steps, ≈ 1 h
scripts/evaluate.sh t0_can_to_metal_bin bc_mlp
```

**Result:** trained (training error 0.68 → 0.055). Success: not yet evaluated.

## 4. ACT — predicting chunks of actions

The same inputs, but LeRobot's ACT predicts the next 100 actions at once.
Same data, same training budget, same yardstick as BC — so the difference
between the two scores is what chunking adds (the ACT paper found it large):

```bash
scripts/train_policy.sh t0_can_to_metal_bin act               # 20k steps, ≈ 1 h 40 min
scripts/evaluate.sh t0_can_to_metal_bin act
```

**Result:** training.

## 5. VLA — adding language and pretraining

LeRobot's SmolVLA also reads the instruction and starts from a model
pretrained on other robots' data; it is fine-tuned on the same demonstrations
(train_evaluation.md §1–2). On T0 the instruction never changes, so this step
tests the pipeline, not language understanding — that needs T1 and later:

```bash
scripts/train_policy.sh t0_can_to_metal_bin smolvla STEPS BATCH
scripts/evaluate.sh t0_can_to_metal_bin smolvla
```

**Result:** not started — STEPS and BATCH to be chosen from a timed run on
this Mac once ACT has finished.

## 8. Deployment (pending)

The last step would take the best learner off the training machine: convert
it to OpenVINO and check it still scores the same. Left out for now; the
export code exists (train_evaluation.md §4).

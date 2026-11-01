# Bimanual SO-101: does the command choose the arm?

A simulated two-arm SO-101 workcell (MuJoCo) where **the command is the only thing that tells the policy what to do**: which task, which arm, and where the cube goes. Nothing in the start scene gives these away, so a policy can only succeed by following the command. The workcell measures how well imitation policies (ACT, Diffusion Policy) and a fine-tuned vision-language-action model do that, including when control passes from one arm to the other.

**Why this matters.** A high success rate doesn't show that a policy read the instruction: if a scene only ever appears with one task, the policy can learn the scene and ignore the command, and LIBERO-Plus found vision-language-action models that scored almost the same with the instruction removed. Recent single-arm benchmarks (InstructMove, LangSwitch) fix this by sharing scenes across tasks, so only the text picks the goal. With two arms the command can also name which arm acts, but in existing bimanual setups the scene usually gives that away: in BiDex-100 the named arm is on its object's side, and in ALOHA and RoboTwin the task or the object's side decides the arm. Here every task can be done by either arm from the same start scenes, and the composites hand control from one arm to the other, so the arm, task and target can only come from the command. Details and references: [0background.md](0background.md).

## Tasks

Six tasks, each doable by either arm from the same start scenes (20 command variants in all):

| | Task | Example command |
|---|---|---|
| S1 | Open drawer | "open the drawer with arm A" |
| S2 | Close drawer | "close the drawer with arm B" |
| S3 | Cube into drawer | "put the cube in the drawer with arm A" |
| S4 | Cube out of drawer | "take the cube out of the drawer and put it on the left of the table with arm B" |
| T1 | Open, then take out (two arms) | "open the drawer with arm A, then take the cube out and put it on the right of the table with arm B" |
| T2 | Put in, then close (two arms) | "put the cube in the drawer with arm B, then close the drawer with arm A" |

## Status

- **Built and validated:** the workcell, a scripted expert for all six tasks (800/800 successful on random starts), randomisation, the recorder, data checks, and policy evaluation in sim.
- **Scope now: the minimal plan.** One scene (drawer closed, cube inside) shared by S1 and T1: 6 command variants, 300 demonstrations, enough to show every point up to a VLA. S2–S4 and T2 are on hold ([1tasks_scene.md](1tasks_scene.md)).
- **Step 1 done:** ACT on "open the drawer with arm A" (50 demonstrations) succeeds and stops in **88% (44/50)** of held-out starts, with ACT's temporal ensembling. All failures are the cube sliding just past the 5 mm limit in the drawer. Training on the noise-free expert commands made no real difference (82%), so the commands as sent are kept. Two findings along the way: a normalisation bug on joints that never move, and temporal ensembling taking the policy from 4/10 to 10/10 ([3train_evaluation.md](3train_evaluation.md)).
- **Steps 2–3 done:** one ACT with a one-hot command input, trained on all six variants (300 demonstrations) of the one scene. **The command decides the task and the target:** from the same start, no S1 rollout went on to take the cube out, all 200 T1 rollouts continued to the hand-over, and none put the cube on the wrong side. T1 succeeds in 97.5% (195/200); S1 in 72% (72/100), every failure the cube sliding 5.0–6.3 mm against the 5 mm limit ([3train_evaluation.md](3train_evaluation.md)).
- **Next (step 4):** SmolVLA fine-tuned on the same data with the text commands, on a rented GPU (~8 days on this Mac, a few hours on an A100 or 4090).

## Quick start

Use the project's `.venv` Python.

```
python -m pytest tests -m "not slow"                                  # fast checks, ~40 s
python -m sim.validate 40                                             # scripted expert on random starts
python -m sim.record data/s1a --variants S1:A --per-variant 50        # record demonstrations (LeRobot dataset)
python -m sim.check_data data/s1a                                     # check a recording
python -m sim.dataset_view data/min data/min_onehot --variant-input   # add the one-hot command input for ACT
python -m sim.evaluate CKPT --variants S1:A --temporal-ensemble 0.01  # evaluate a trained policy in sim
```

Training uses LeRobot 0.6.1 (`lerobot-train`); the runs behind the results are in `scripts/`. `cloud/train.sh` runs ACT or SmolVLA on a rented GPU: [cloud/README.md](cloud/README.md).

## Documents

| File | Contents |
|---|---|
| [0background.md](0background.md) | Motivation, related work, the research question, what is evaluated, scope |
| [1tasks_scene.md](1tasks_scene.md) | Tasks, scene, randomisation, the scripted expert, success checks, files, and the step-by-step plan |
| [2data_collection.md](2data_collection.md) | Recording design, data checks, cube colours |
| [3train_evaluation.md](3train_evaluation.md) | Training and evaluation results, bugs and findings |
| [cloud/README.md](cloud/README.md) | Step by step: training on a rented GPU, evaluating on the Mac |

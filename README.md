# Bimanual SO-101: does the command choose the arm?

A simulated two-arm SO-101 workcell (MuJoCo) where **the command is the only thing that tells the policy what to do**: which task, which arm, and where the cube goes. Nothing in the start scene gives these away, so a policy can only succeed by following the command. The workcell measures how well imitation policies (ACT, Diffusion Policy) and a fine-tuned vision-language-action model do that, including when control passes from one arm to the other.

**Why this matters.** A high success rate doesn't show that a policy read the instruction: if a scene only ever appears with one task, the policy can learn the scene and ignore the command, and LIBERO-Plus found vision-language-action models that scored almost the same with the instruction removed. Recent single-arm benchmarks (InstructMove, LangSwitch) fix this by sharing scenes across tasks, so only the text picks the goal. With two arms the command can also name which arm acts, but in existing bimanual setups the scene usually gives that away: in BiDex-100 the named arm is on its object's side, and in ALOHA and RoboTwin the task or the object's side decides the arm. Here every task can be done by either arm from the same start scenes, and the composites hand control from one arm to the other, so the arm, task and target can only come from the command. Details and references: [docs/0background.md](docs/0background.md).

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
- **Scope now: the minimal plan.** One scene (drawer closed, cube inside) shared by S1 and T1: 6 command variants, 300 demonstrations, enough to show every point up to a VLA. S2–S4 and T2 are on hold ([docs/1tasks_scene.md](docs/1tasks_scene.md)).
- **Step 1 done:** ACT on "open the drawer with arm A" (50 demonstrations) succeeds and stops in **88% (44/50)** of held-out starts, with ACT's temporal ensembling. All failures are the cube sliding just past the 5 mm limit in the drawer. Training on the noise-free expert commands made no real difference (82%), so the commands as sent are kept. Two findings along the way: a normalisation bug on joints that never move, and temporal ensembling taking the policy from 4/10 to 10/10 ([docs/3train_evaluation.md](docs/3train_evaluation.md)).
- **Steps 2–3 done:** one ACT with a one-hot command input, trained on all six variants (300 demonstrations) of the one scene. **The command decides the task and the target:** from the same start, no S1 rollout went on to take the cube out, all 200 T1 rollouts continued to the hand-over, and none put the cube on the wrong side. T1 succeeds in 97.5% (195/200); S1 in 72% (72/100), every failure the cube sliding 5.0–6.3 mm against the 5 mm limit ([docs/3train_evaluation.md](docs/3train_evaluation.md)).
- **Held-out cube colours:** no gap. Step 1's policy keeps its 88% with colours it never saw; step 3's gets S1 77/100 and T1 194/200 (against 72 and 195), and T1 is where colour matters, since the second arm has to find and grasp the cube.
- **Now (steps 4 and 3b):** SmolVLA fine-tuned on the same data with the text commands (trained on a rented RTX 4090 in ~5 h, ~8 days on this Mac; evaluated: level with ACT on S1, T1 56% against 98% from loose placement, no colour gap, and it breaks under an unseen sentence shape: T1 0/200), and Diffusion Policy with the one-hot input, trained there for 100k steps (a 5-rollout check: S1:A 4/5 with the other arm still, T1:A:left 5/5; full evaluation queued, with 10 DDIM denoising steps instead of 100). Live state: [docs/STATUS.md](docs/STATUS.md).
- **Next (step 5): stress tests.** ACT, DP and SmolVLA under one factor at a time pushed past its training range (more distractors, a wider arm start pose, the cube turned further), as The COLOSSEUM and LIBERO-Plus test policies: does the VLA's pretraining buy robustness? Residual RL is parked as an optional step 6, for a gap the stress tests find (steps 1–3 leave it nothing real to fix: S1's failures are slides just past a 5 mm limit).

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

Training uses LeRobot 0.6.1 (`lerobot-train`); the runs behind the results are in `scripts/`. `cloud/train.sh` runs ACT, SmolVLA or DP on a rented GPU: [docs/4cloud_training.md](docs/4cloud_training.md).

## Documents

| File | Contents |
|---|---|
| [docs/STATUS.md](docs/STATUS.md) | What is running now, what's next, open questions |
| [docs/0background.md](docs/0background.md) | Motivation, related work, the research question, what is evaluated, scope |
| [docs/1tasks_scene.md](docs/1tasks_scene.md) | Tasks, scene, randomisation, the scripted expert, success checks, files, and the step-by-step plan |
| [docs/2data_collection.md](docs/2data_collection.md) | Recording design, data checks, cube colours |
| [docs/3train_evaluation.md](docs/3train_evaluation.md) | Training and evaluation results, bugs and findings |
| [docs/4cloud_training.md](docs/4cloud_training.md) | Step by step: training on a rented GPU, evaluating on the Mac |

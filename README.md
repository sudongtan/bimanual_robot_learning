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
- **Now:** plan step 1, ACT on "open the drawer with arm A" (50 demonstrations), trained on the commands with and without demonstration noise.
- **No policy results yet.**

## Quick start

Use the project's `.venv` Python.

```
python -m pytest tests -m "not slow"                                  # fast checks, ~40 s
python -m sim.validate 40                                             # scripted expert on random starts
python -m sim.record data/s1a --variants S1:A --per-variant 50        # record demonstrations (LeRobot dataset)
python -m sim.check_data data/s1a                                     # check a recording
python -m sim.evaluate CKPT --variants S1:A --condition train         # evaluate a trained policy in sim
```

## Documents

| File | Contents |
|---|---|
| [0background.md](0background.md) | Motivation, related work, the research question, what is evaluated, scope |
| [1tasks_scene.md](1tasks_scene.md) | Tasks, scene, randomisation, the scripted expert, success checks, files, and the step-by-step plan |
| [2data_collection.md](2data_collection.md) | Recording design, data checks, cube colours |
| [3train_evaluation.md](3train_evaluation.md) | Training and evaluation results, bugs and findings |

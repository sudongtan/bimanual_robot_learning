# Bimanual SO-101: does the command choose the arm?

A simulated two-arm SO-101 workcell (MuJoCo) where **the command is the only thing that tells the policy what to do**: which task, which arm, and where the cube goes. The start scenes are designed and audited so that none of these can be read off the scene. The workcell is used to measure how well imitation policies (ACT, Diffusion Policy) and a fine-tuned vision-language-action model follow commands, including when control passes from one arm to the other.

*Status: the workcell, scripted expert and randomisation are built and validated. Data recording and training are next; no policy results yet.*

## Purpose

A policy's success rate doesn't show whether it read the instruction. If a scene only ever appears with one task, the policy can learn "this scene means that task" and ignore the command. [LIBERO-Plus](https://arxiv.org/abs/2510.13626) found exactly this: some vision-language-action models scored almost the same with the instruction removed. Recent benchmarks answer by making the text indispensable. [InstructMove](https://arxiv.org/abs/2608.22990) puts several feasible goals in each scene so only the instruction picks one, and [LangSwitch](https://arxiv.org/abs/2608.14822) (introduced with the CoRe recovery method) shares one scene across skills and switches instructions mid-episode.

These are single-arm. With two arms there is one more thing the command can specify: **which arm acts**. Bimanual benchmarks such as [BiDex-100](https://arxiv.org/abs/2610.25459) and Bimanual RLBench write arm-specific instructions, but an arm-specific instruction only tests command-following if the scene can't also give the arm away. In BiDex-100 the named arm is on the same side as its object ("with the left arm use the left hammer"). In ALOHA and RoboTwin, which arm acts is usually fixed by the task or by which side the object is on.

This project asks:

> **When nothing in the scene says which task, which arm or which target, how well do policies follow the command, and does that hold when control passes from one arm to the other?**

To make the command the only signal:

- **The command names the arm.** Every task can be done by either arm, from the same start scenes.
- **Every start scene is shared** by at least two tasks.
- **Item starts never depend on the target.** Every start-state factor was audited for any dependence on task, arm or target.

The tasks are four primitives (open the drawer, close it, put the cube in, take it out to the left, middle or right of the table) and two composites that hand over between arms: one arm opens the drawer and the other takes the cube out, or one arm puts the cube in and the other closes the drawer.

The same demonstrations train two kinds of policy:

- **ACT and Diffusion Policy with a one-hot input** for task, arm and target, the way Meta-World MT10/MT50 adds a task ID. The policy is told exactly what to do.
- **A fine-tuned vision-language-action model** (SmolVLA, then π0 if compute allows), which has to read the command in natural language.

Both are evaluated with unseen colours, and the VLA also with unseen wording, each change measured on its own, plus:

- **Swap tests:** keep the scene, change only the task, arm or target in the command, and check that the behaviour follows the command.
- **Stop tests:** after "open the drawer", the scene is the start of "take the cube out". Does the policy stop because of the command, or carry on because of the scene? Every demonstration ends with a recorded 2 s hold at home, so stopping is in the training data, and rollouts run past the demonstration length so a policy that carries on is caught.
- **Hand-over tests:** run a composite as two primitive commands in a row, switching arms between them, and see how much success is lost at the switch. For T1 the second command starts from a scene it was trained on; for T2 it doesn't (the drawer is partly open with the cube inside, which no close-the-drawer demo starts from), so T2's loss also includes an unseen start.

## Scope

- **The arms act in turn, never at the same time.** The composites pass control from one arm to the other; there is no simultaneous coordination such as a hand-off.
- **The demonstrations are scripted** from simulator state, so they are cleaner and more consistent than human teleoperation.
- **This is a small, controlled testbed** (six tasks, 20 command variants), not a benchmark comparable to LIBERO or RoboTwin.
- **Simulation only.**

The tasks, scene and design decisions are in [1tasks_scene.md](1tasks_scene.md); the recording design is in [2data_collection.md](2data_collection.md).

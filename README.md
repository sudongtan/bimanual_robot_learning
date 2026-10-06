# Goal

Build an end-to-end bimanual robot learning system in MuJoCo:

**Natural language + camera observations → learned policy → coordinated SO-101 actions → waste sorted at the table**

Main progression:

**Scripted expert → Behavioral Cloning → ACT → VLA → optional RL → OpenVINO deployment**

# Task

## The story: sorting waste where it is made

Dinner is over. The table holds a mix of things that all leave by different
routes: a half-finished bottle of water, an empty drink can, a mug with some
coffee left in it, a plate with food scraps, used cutlery.

Recycling only works if each item goes into the right stream, and goes in
empty and clean enough to be processed. A bottle with liquid still in it, a can
dropped in the plastic bin, or a fork thrown away with the rubbish all undo
the effort further down the line. The easiest place to get it right is the
table itself, before anything is mixed.

That is the robot's job. Two SO-101 arms mounted at the edge of the table
(arm A on the left, arm B on the right) take each item, empty it if needed,
and send it to the right waste stream.

## Where everything goes

| Waste stream | Destination |
|---|---|
| Plastic recycling | plastic bin (under-table pull-out drawer) |
| Metal recycling | metal bin (under-table pull-out drawer) |
| Food waste | food-waste container on the table |
| Reuse (to wash) | "to wash" basket on the table |

## Tasks

Arm A is the left arm, arm B the right arm. Implementation notes per task:
[docs/tasks.md](docs/tasks.md).

| ID | Instruction | Success |
|---|---|---|
| T1 `can_to_metal_bin` | "Pick up the tomato soup can with arm A, open the metal bin with arm B, and put the can in the metal bin with arm A." | the can is in the metal bin |
| T2 `cup_to_plastic_bin` | "Pick up the plastic cup with arm B, open the plastic bin with arm A, and put the cup in the plastic bin with arm B." | the cup is in the plastic bin |
| T3 `handoff_can_to_metal_bin` | "Pick up the tomato soup can with arm A, hand it to arm B, and put it in the metal bin with arm B." | the can is in the metal bin, handed over without touching the table |
| T4 `empty_bottle` | "Hold the food-waste container with arm A, and pour the leftover water from the bottle into it with arm B." | the water ends up in the container |
| T5 `mug_to_wash_basket` | "Put the mug in the to-wash basket." | the mug is in the basket |
| T6 `clear_the_table` | "Clear the table." | every item is in its correct destination |

## Phases

1) Build scenes with randomization and the Bimanual MuJoCo Environment
2) Generate Expert Demonstrations
3) train and evaluate behaviour clone baseline
4) train and evaluate ACT
5) finetune and evaluate VLA
6) Optional: train and evaluate diffusion policy
7) Optional: reinforcement learning
8) OpenVINO deployment

## Steps
1) T1 and phase 1, 2, 3, 4, 5, 8
progress: 
phase 2 is only half done. The smoke test showed that the expert works (50/50), but it didn't record anything. Phase 2 means generating demonstrations, a saved dataset, and none exists yet.

Phase (your README)	T1 status
1 Scene with randomization + environment	done for T1 (only the can's position and rotation are randomized)
2 Expert demonstrations	expert done, recording not started
3 BC, train + evaluate	not started
4 ACT	not started
5 VLA	not started
8 OpenVINO	not started
To finish phase 2: run the expert through the environment for about 50 seeds and save each successful episode in LeRobot's dataset format. That's what LeRobot's BC and ACT training reads. Each recorded step would contain:

camera images;
the 12 joint positions (the state);
the 12 joint targets (the action);
the task instruction;
the seed and a success flag.

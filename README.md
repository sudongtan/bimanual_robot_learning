# Goal

Build an end-to-end bimanual robot learning system in MuJoCo:

**Natural language + camera observations → learned policy → coordinated SO-101 actions → waste sorted at the table**

Main progression:

**Scripted expert → Behavioral Cloning → ACT → VLA → diffusion -> optional RL → deployment**

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
| Plastic recycling | plastic bin (under-table pull-out bin) |
| Metal recycling | metal bin (under-table pull-out bin) |
| Food waste | food-waste container on the table |
| Reuse (to wash) | "to wash" basket on the table |

## Tasks

Arm A is the left arm, arm B the right arm. Implementation notes per task:
[docs/tasks_data.md](docs/tasks_data.md).

| ID | Instruction | Success |
|---|---|---|
| T0 `t0_can_to_metal_bin` — T1 with only the can on the table (pipeline smoke test) | "Pick up the tomato soup can with arm A, open the metal bin with arm B, and put the can in the metal bin with arm A." | the can is in the metal bin |
| T1 `can_to_metal_bin` | "Pick up the tomato soup can with arm A, open the metal bin with arm B, and put the can in the metal bin with arm A." | the can is in the metal bin |
| T2 `cup_to_plastic_bin` | "Pick up the plastic cup with arm B, open the plastic bin with arm A, and put the cup in the plastic bin with arm B." | the cup is in the plastic bin |
| T3 `handoff_can_to_metal_bin` | "Pick up the tomato soup can with arm A, hand it to arm B, and put it in the metal bin with arm B." | the can is in the metal bin, handed over without touching the table |
| T4 `empty_bottle` | "Hold the food-waste container with arm A, and pour the leftover water from the bottle into it with arm B." | the water ends up in the container |
| T5 `mug_to_wash_basket` | "Put the mug in the to-wash basket." | the mug is in the basket |
| T6 `clear_the_table` | "Clear the table." | every item is in its correct destination |

## Phases
Preparation
1) Build scenes with randomization and the Bimanual MuJoCo Environment
2) Generate Expert Demonstrations and prepare for evaluation

Training and evaluation
3) train and evaluate behaviour clone baseline
4) train and evaluate ACT
5) finetune and evaluate VLA
6) Optional: train and evaluate diffusion policy
7) Optional: reinforcement learning

Deployment 
8) OpenVINO deployment — pending (deployment left out for now)

## Progress
1) Smoke test: T0 and phase 1, 2, 3, 4, 5 

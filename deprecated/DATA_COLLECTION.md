

Totally we need 400 episodes
200 for picking up can, 100 from left arm, 100 from right arm.
200 for picking up cube, 100 from left arm, 100 from right arm.

## Randomisation per episode

| What | How | Where in code |
|---|---|---|
| Item start position | Uniform inside the four start patches: \|x\| 8–14 cm on the picking arm's side, \|y\| 16–20 cm front or back (README 2.4) | `sim/region_map.py` `sample_start` |
| Arm start pose | Gaussian noise on all 5 arm joints of both arms, σ = 0.02 rad (fingertip moves ~0.5–1 cm) | `sim/scene.py` `START_NOISE`, `Sim.reset` |
| Demonstration noise | Smooth random offset on the arm joint commands, σ = 0.008 rad, correlation time 0.25 s. At the fingertip: about 3 mm RMS, 95% under 5.6 mm, max under 1 cm. | `sim/expert.py` `DEMO_NOISE`; `sim/scene.py` `Sim.noisy` |

**Demonstration noise** follows ACT's scripted-data option (`inject_noise`, ±1 cm on gripper targets) but at about half that size, so the demos include small corrections without risking failed episodes.
- **Free-space moves only:** approaching above the pre-grasp, lifting, carrying, the opener's approach to the handle, and returning home.
- **Off for the precise parts:** going in to grasp, closing, the handle grasp and pull, the release over the drawer. The noise fades out (0.2 s) and the arm settles before each of these, so they start on target.
- **Validated:** with the noise on, 160/160 expert episodes from random starts succeed (80 per item, both arms): `python -m sim.region_map validate --n 80`.
- **When recording,** save the commands the actuators actually received (`Sim.applied_ctrl`, which includes the noise) as the actions, not the noise-free plan. Then the actions match what the observations show.

**Fixed (not randomised):** block rotation, arm base positions, cameras, lighting, drawer starting state (closed), item size and mass.

**Arm balance:** `sample_start` picks the side at random (an earlier run gave 33/47). For the 100/100 split per item, draw the side deliberately, e.g. `sample_start(rng, arm="left")` for half the episodes and `arm="right"` for the rest, and alternate front and back patches.

in the scene setting: 
1) two things are randomized.
2) the color of the items should be randomized
later in recording: lighting 

what's on the table: one can, one cube, one - two distractors.  when can is the target item, cube should also be in a region that is proposed for targets, so you cant tell which one is the target based on location. the distractors are totally randomized.

task 1a and 1b are trained separately.

finetune scripted expert step by step 
task 1a: 25 can with left arm, 25 can with right arm together for each step, stop at 100 episode. for the next 100 episodes, run it, and only review when the success rate drops to 90%.










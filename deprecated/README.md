The purpose of this project is to set up a simulation in MuJoCo and collect data
for training ACT, diffusion, and VLA models.

# 1. What is given

- **Two SO-101 arms** in MuJoCo.
- **One table with drawers and items** for the arms to work with.
- **Basic tasks:**
  1. One arm opens a drawer while the other picks up an item and puts it in the drawer. The arm that is physically "closer" to the item will pick up the item and the other arm open the draw. An ACT/Difusion model/(Reinforcement learning) is going to learn that. (there are two target items so two subtasks.)
  2. VLA pick-and-place instructions, given the scene, and instruction “Put the plastic cube in the plastic bin” and “Put the can in the metal bin.” , the closer arm will pick up the item and the other arm open the bin. so there will be at least two physically distinguishable items. 
  3. One arm hands an item to the other arm. (Optional)

# 2. Scene

## 2.1 Requirements

The scene must satisfy:
A. Two arms:
- The two arms dont interfere with each other.
- A handover between the two arms is possible.

B. Bins:
- Both arms can open both bins and the opened gap is bigger than the biggest target item.

C. Target Items
- when an item is obviously at a position that is reachable to the arm, the item can be picked up by an arm and does not drop while the arm moves to the side of the table. 

D. where should the target item be on the table
- The items is reachable by at least one arm. 
- The items dont sit exactly in the middle of two arms.
- Chosen region: section 2.4.

E. (Optional) The item should be visible to a camera once it is in the drawer.


## 2.2 Current scene

A basic scene has been built and tested. The values below are read from `sim/scenes/sorting_table.xml`, the loaded model and the task file.

**Coordinates:** the table centre is at x = y = 0. +x points toward the right arm, +y toward the back. The tabletop is at z = 0.315 m. All positions are in metres.

### Table

- 0.50 × 0.46 m (x from −0.25 to 0.25, y from −0.23 to 0.23)
- Top at z = 0.315 m, four legs

### Arms

Two SO-101 arms (five joints plus a gripper each, 12 actuators in total), sitting on the tabletop on the centre line, 5 cm in from the short edges. They face each other, 0.40 m apart.

| | Left arm (A) | Right arm (B) |
|---|---|---|
| Base position | (−0.20, 0) | (+0.20, 0), turned 180° |
| Home fingertip position | (−0.168, 0, 0.689) | (0.168, 0, 0.689) (mirrored) |

- **Home pose:** each arm stands upright over its own base, gripper pointing toward the other arm and about 35° up (joints `[-0.04, -1.51, -0.42, 1.34, 0.11]`, all ≥ 0.23 rad inside their limits). Closest gap between the arms: 22 cm.
  - Replaces the earlier diagonal home (fingertip at (−0.072, 0.117, 0.449)). That pose leaned each arm over the drawer the *other* arm delivers to and caused arm collisions. A centre-line home is out of reach (shoulder_lift hits its limit).
- **Start-pose noise:** σ = 0.02 rad per joint, sampled each episode.

### Cameras

| Camera | Mount | Field of view | Notes |
|---|---|---|---|
| Overview | Fixed, looking straight down from (0, 0, 1.25) | 45° | Image right = +x (right arm), image up = +y (plastic drawer) |
| Wrist (×2) | One on each arm | 75° | |

Training images are 224 × 224.

### Bins (drawers under the table's long sides)

| | Metal | Plastic |
|---|---|---|
| Side | Front, y < 0 | Back, y > 0 |
| Interior centre (closed) | (0, −0.12, 0.22) | (0, +0.12, 0.22) |
| Interior size | 15 × 19 × 12 cm | 15 × 19 × 12 cm |
| Handle bar | (0, −0.285, 0.395) | (0, +0.285, 0.395) |
| Pull | 12 cm toward −y | 12 cm toward +y |

- Both drawers are centred in x, so they are equally far from both arms.
- The handles stand 8 cm above the tabletop and 5.5 cm out from the drawer fronts.

### Items (2 or 3 per table)

| Item | What it is | Size | Mass | Role |
|---|---|---|---|---|
| Can | Small metal can (cylinder) | 4.0 cm wide, 6.0 cm tall | 80 g | Metal item; target |
| Block | Plain box (`block30x50`) | 3 × 3 × 5 cm | 9 g | Plastic item; target |
| Lemon or banana | YCB | Banana ≈ 18 cm long | — | Distractor: 0 or 1 per table, no bin |

The can was the YCB tomato soup can (6.8 × 10.2 cm, 349 g). With a real approach, the SO-101 gripper couldn't pick it reliably at any spot. A side grasp swept over the rim and pushed it about 3.5 cm; a grasp from above was out of reach or didn't hold. Size sweep (Task 1 expert, 12 spots, grasped from above at 20°): 4.0 × 6.0 cm 11/12, 4.5 × 6.5 cm 9/12, 5.0 × 7.0 cm 9/12, 5.5 × 8.0 cm 4/12.


## 2.3 Simulation build and requirement check

Built from section 2.2:

| File | What it is |
|---|---|
| `sim/scenes/sorting_table.xml` | The scene: table, both drawers, items, cameras, two attached SO-101 arms |
| `sim/assets/so101/` | SO-101 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (Apache-2.0); wrist camera set to 75° |
| `sim/scene.py` | Loading, home pose, IK, scripted controller, arm-to-arm contact monitor |
| `sim/check_requirements.py` | Runs requirements A–E in simulation and prints a report |
| `sim/expert.py` | Scripted expert for Task 1 (reads item poses from the simulator, one fixed grasp per item, continuous motion from home) |
| `sim/region_map.py` | Start-region maps (`ik`, `expert`), the chosen region (`X_RANGE`, `Y_RANGE`, `sample_start`) and its validation (`validate`) |
| `tests/test_scene.py` | Fast checks plus runs per requirement and expert episodes from the start region (`-m 'not slow'` skips the runs) |

Differences from section 2.2:
- The can, block and banana are simple shapes (cylinder, box, capsule) with the listed sizes and masses; there are no YCB meshes yet. The banana has a placeholder mass of 120 g. The lemon isn't modelled.
- Each drawer handle is a goalpost standing outside the table edge, so the bar can sit 8 cm above the tabletop as specified.
- Drawers open about 10.7 cm, not the full 12 cm. With the bar held 1 cm into the jaws the arm can follow about 11.5 cm, so the pull aims 1 cm short.

Results (`python -m sim.check_requirements`, 3 seeds; scripted motions). The C and Task 1 rows of that report teleport the gripper into the grasp pose and use spots outside the start region. The expert validation in the next section is the authoritative pick result.

| Req. | Result |
|---|---|
| A1 arms don't interfere | Home: 50/50 noise seeds with no contact, closest gap 22.1 cm. Expert episodes from the start region: no arm contact in 160/160. In Task 1 the opening arm goes home (via a point on its own side) before the other arm picks. The broader report has arm contact in 3 of its 90 scripted runs, all in its teleported Task 1 rows at spots outside the start region. |
| A2 handover | 6/6 (both directions): the giver brings the block to (0, 0.08, 0.42), the taker grips its lower half from the side at 45° down. On the centre line (y = 0) the taker would need joints at their limits. |
| B both arms open both drawers | 12/12 runs (4 arm–drawer pairs × 3 seeds) open about 10.7 cm and stay open. |
| C pick up and carry | Expert, full approach from home: 160/160 from the start region (next section). The broader report's teleported grasps, over spots across each arm's half: block 20/26 and can 25/26 still held at the drawer. |
| D reachable | Every tabletop grid cell is reachable by at least one arm, and 77% by both. With a fixed grasp and a 6 cm pre-grasp, though, the middle strip (|y| below about 10–12 cm) can't be picked (next section). |
| E visible in drawer | Expert drops from the start region, 20 per item, 224×224 images: the block was visible from the overview camera every time (≥ 82 px). The can was invisible to all three cameras in 1 of 20 drops, when it rolled under the tabletop part of the drawer. |


## 2.4 Where items start on the table

**Start region: four 6 × 4 cm patches,** the same for both items:

```
  y=+0.20  ┌────┐                ┌────┐
   back    │ L  │                │ R  │
  y=+0.16  └────┘                └────┘
         x=-0.14..-0.08     x=+0.08..+0.14
  y=-0.16  ┌────┐                ┌────┐
   front   │ L  │                │ R  │
  y=-0.20  └────┘                └────┘
  arm bases at (∓0.20, 0); L/R = picking arm (the closer one)
```

- **Definition:** |x| 8–14 cm on the picking arm's side, |y| 16–20 cm in front or behind (`X_RANGE`, `Y_RANGE`, `sample_start` in `sim/region_map.py`). That's 48 cm² per arm (96 cm² in total, about 4% of our small table). In absolute terms this is **smaller than the references**: about half of ManiSkill's SO-100 setup (10 × 10 cm = 100 cm², same arm size class), and about 1/8–1/10 of ACT (20 × 20 cm) and RoboTwin (20 × 25 cm), which use larger arms. The main loss is the middle strip, which the current expert can't pick from; see below.
- **The expert that defines it** (`sim/expert.py`) works like the reference experts (ACT, ManiSkill, RoboTwin): it reads item poses from the simulator and uses one fixed grasp per item. Both items are grasped from above: the block tilted 10° outward, the can 20°.
  - The opener goes via a point on its own side to the handle, opens the drawer and returns home.
  - The picker moves in joint space to a point above its pre-grasp, then descends straight to the pre-grasp (6 cm back along the gripper), goes straight in and closes. It lifts, carries in a straight line with the gripper's pointing direction controlled (turning gradually to 30° from vertical), drops into the drawer and returns home.
- **Validation:** `python -m sim.region_map validate --n 80` gives **160/160** episodes succeeding: item in the drawer, still held when carried there, no arm contact, nothing pushed during the approach. Uniform random starts, both items, both arms, random start noise and demonstration noise (σ = 0.008 rad on free-space moves; see `DATA_COLLECTION.md`); largest joint jump 0.31 rad. Zero failures in 80 runs per item means at least 96% success with 95% confidence.
- **Why these bounds** (from `python -m sim.region_map expert`, a 2 cm grid):
  - **|y| below about 10–12 cm** (depending on x) fails for both items: too close to the arm's own base for the pre-grasp.
  - **|y| > 20 cm** puts the item within 1 cm of the table edge.
  - **x 6–8 cm:** an earlier validation with |x| 6–14 cm had 3 of 60 block starts fail, all at x 6–7 cm, |y| about 16.5 cm.
  - **|x| ≥ 16 cm** fails for the can on the back band (and |x| ≥ 18 cm for the block on the front band).
- **Leakage:** both items use the same patches, so start position doesn't reveal which item it is.
- **Still open:**
  - Which arm picks follows from the side, but random sampling gave 33/47 left/right. For data collection, alternate sides (or stratify) so both arms pick equally often.
  - Block rotation is fixed (faces aligned with the table).
  - Task 2 needs both items at once, in different patches. Not validated yet.

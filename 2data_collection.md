
## Next step: recording

The recorder is written (`sim/record.py`). The step-0 batch (A0, `data/a0`, 5 per variant) is recorded and checked: 100/100 successful, no action or label flags, but the check showed some cube colours blending into the scene, so the palette was replaced ("Cube colours" below). A0 is not used for training (step-0 batches are discarded; its seeds are recorded again in the full set), and it uses the old colours anyway; all training data is recorded fresh with the new palette. Next: the training pilot (plan step 1).

```
python -m sim.record data/pilot --per-variant 5                  # step 0: 5 of each of the 20 variants
python -m sim.record data/s1a --variants S1:A --per-variant 50   # plan step 1
```

**Hook.** Every `run_*` takes `rec`, called as `rec(sim)` after every 2 ms physics step during the task. The expert runs at 50 Hz (`CONTROL_EVERY` = 10 steps, zero-order hold, design point 1): on the call after the first step of each 20 ms interval `sim.ticked` is true, and `sim.tick_state` (the physics state before the interval, `STATE_SPEC`), `sim.applied_ctrl` (the command held over it, with demonstration noise), `sim.nominal_ctrl` (the same without noise) and `sim.subtask` are one frame. Settling steps before the task starts are not passed to `rec`. The 2 s hold at the end (`HOLD_S`, in `finish`: both arms at home, nothing moving) is, so every episode demonstrates staying still after the task (the stop tests in `1tasks_scene.md`). Episode lengths including the hold (one episode each, 50 fps frames): S1 10.5 s (524), S2 9.9 s (494), S3 12.9 s (644), S4 11.7 s (586), T1 20.4 s (1,020), T2 21.1 s (1,056).

**Recorder (`sim/record.py`):**
- One frame per control tick (50 fps, as ACT's sim data; see the frame-rate note below).
- Two passes. The expert walks each variant's seeds in parallel and saves every episode's frames. Then the successful episodes, in order, are split into one chunk per worker; each worker rebuilds the episode's model from its seed (colours and lighting are drawn from it), **checks the replay is exact** (first-batch check 1, on every episode), renders the three cameras from the recorded states and writes a small dataset; the chunks are merged in order with LeRobot's `aggregate_datasets`, so episode indices don't depend on the number of workers.
- LeRobot dataset, per frame: `observation.state` (the 12 actuated joints, with the actuator names in the feature metadata), `action`, `action_nominal`, `observation.images.{overview,left_wrist,right_wrist}` (480 × 640, video), `subtask_index`, `variant_index` (into `VARIANTS`, to build the one-hot input). The episode's task is its command. ACT, SmolVLA and π0 load it directly.
- **Failed episodes are kept but not in the LeRobot dataset**, so it trains as is (only successful episodes): they are in `sim/episodes.jsonl` and `sim/states/`, and can be rendered later if needed.
- `sim/episodes.jsonl`: every episode run, with its variant, seed, the `run_*` result (success, command, `scene` with the drawn colours and lighting, checks) and `episode_index` in the dataset. `sim/states/*.npz`: per episode, the float64 physics state before each frame (`STATE_SPEC`), both actions and the subtask, enough to replay exactly or re-render any camera (LeRobot reads float columns back as float32, too coarse for exact replay). `sim/info.json`: git commit (and whether `sim/` had uncommitted changes), control rate, randomisation settings.
- **Speed:** rendering is ~95% of the time, ~15 ms per camera image whatever the resolution, so it is the triangle count: the Menagerie SO-101 meshes are ~700k triangles for both arms, drawn twice (the shadow pass halves the time when off). The GPU is shared, so parallel workers gain only ~10%: ~70 images/s, i.e. ~50 min for the 100-episode pilot and ~8 h for the 1,000. Options if that matters: decimate the visual-only meshes (collision and physics unchanged, so replay stays exact), or record states only (`--no-images`) and render later.
- On this Mac `torchcodec` can't load (no FFmpeg shared libraries), so LeRobot decodes video with `pyav`; encoding is unaffected. Install FFmpeg (e.g. `brew install ffmpeg`) before training locally.
- Seeds (`sim/randomize.py`). A seed fixes the whole start, so the same seed gives the same start in every task that shares the scene. **Training: a unique seed per episode**, so all 1,000 recorded starts differ (as LIBERO keeps separate init states per task and RoboTwin draws each episode); with one seed range shared by all variants a scene would have only ~50 distinct starts. Variant k records from its own block of 500 seeds, `train_seeds(task, arm, target)`, within `TRAIN_SEEDS` (1–10,000). **Evaluation: the same `EVAL_SEEDS` (from 100,000) for every variant**, so swap tests compare commands on identical starts, with `Randomization(split="eval")` for the held-out conditions. A failed episode is kept, flagged, and the next seed in the block is used. S2's start kind is drawn from the seed (seeds 1–20: 12 partly open, 8 cube inside), so the recorder walks the block until each kind has its 25. The step-0 pilot uses the start of each block; it is discarded, so those seeds are re-recorded in the full set.
- One LeRobot dataset for all tasks, with the command as each episode's task (fits the VLA plan; ACT can train on a filtered subset).

**Design points:**

1. **Action timing: hold each command for one frame. Done.** A 50 fps sample of a 500 Hz expert would skip 9 commands in between and replay could drift, so the expert runs at 50 Hz, as ACT's scripted sim policy runs at its control rate: `Sim.step` latches `ctrl` every `control_every` steps (`new_sim` sets `CONTROL_EVERY` = 10) and the demonstration noise advances once per interval. A frame is (state before the 10 steps, command held during them). `test_replay_is_exact` replays S1 and T1 from the recorded states and commands and matches every recorded state exactly.
   - Side effect: the coarser drawer pull slid the cube in the drawer past S1's 5 mm limit (6–7 mm). Most of it (4 mm, also at 500 Hz) was the gripper snapping shut on the handle, which drags the drawer ~5 mm in 20 ms; the handle grip now closes gradually (`grip(..., ramp=True)`), leaving 1.6–2.9 mm at 50 Hz.
2. **Noisy and clean action labels: store both.** `action` = the command sent, including demonstration noise (as ACT records it). `action_nominal` = the expert's command without noise. DART (Laskey et al. 2017) runs the noisy commands but trains on the clean ones: they are position targets, so they point back to the planned path and act as recovery labels. Pick which to train on by comparing both in the pilot.
3. **Frame rate: 50 fps.** Matches ACT's sim data and π0, and divides the 2 ms timestep evenly (one command per 10 steps). Training and evaluation are both in sim, so there is no real-robot rate to match.
4. **Image resolution: record at 480 × 640.** With the overview at 224 × 224 (and its old 45° view, mostly floor), the cube was ~8 px across. The overview now has a 30° field of view, framing the table: the cube is ~40 px at 480 × 640 and ~14 px resized to 224. ACT rendered at 480 × 640, and LeRobot stores video and resizes in the policy, so record at 480 × 640 (or re-render later from `qpos` + `scene`) and let each policy resize.
5. **Composite sub-commands as per-frame labels. Hook done:** `sim.subtask` is 0, then 1 from the moment the first arm is back home (`test_composite_subtask_boundary`). A LeRobot episode has one `task` string, so keep the full T1/T2 command as the task and add a per-frame `subtask_index` (T1: 0 = S1 part, 1 = S4 part; T2: 0 = S3 part, 1 = S2 part) plus the frame where the first arm is back home. Don't cut a composite into two episodes: the second half wouldn't start from home. The boundary is also where the chained-primitives evaluation switches command (plan step 13).
6. **Balance per command, not just per task × arm.** S1, S2, S3 and T2 have 2 commands each (arm, or direction); S4 and T1 have 6 (arm × target). Equal counts per task × arm leave each S4 command with a third of the episodes of an S1 command. Default: equal episodes per command, plus both start kinds of S2.

**Checks on the first batch** (`python -m sim.check_data data/a0`: check 1 runs in the recorder on every episode; this script runs 2–4 on the LeRobot dataset and writes `check/report.txt`, with every threshold breach listed under FLAGS, `check/episodes.json` and one contact sheet per episode in `check/sheets/`):
1. **Replay:** play the recorded actions in a fresh sim from the recorded start state (each held for 10 steps) and confirm every recorded state matches exactly (as `test_replay_is_exact`) and the task still succeeds.
2. **Images:** the cube is visible and not hidden by the arms; the wrist cameras see the cube, handle and drawer inside when it matters; colours and lighting look right.
3. **Actions:** no jumps (watch the carry to the far-side S4/T1 targets), no long idle stretches, sensible gripper timing.
4. **Labels:** the command matches what happens and the named arm is the one moving.

**Data notes:**
- T1 episodes are also S1 + S4 demos, T2 episodes S3 + S2: mark the boundary and label each part with its sub-command (design point 5).
- Balance episodes per command, both directions of each composite, and both start kinds of S2 (design point 6).

**Open:**
- Episode counts. Pilot: 25 per task and arm. Default for the full set: 50 per command (ACT used 50 sim demos per task; LIBERO has 50 per task, the set SmolVLA and π0 are fine-tuned on).
- Automated render check: done in `sim/check_data.py`: cube pixels at 480 × 640 and at 224, and contrast with each surround, with the cube matched by its segmentation ID (lighting changes the rendered colour too much for a colour match).

**Cube colours (measured on A0).** A0's check flagged 9 of 100 episodes, all orange or yellow cubes, as low contrast with the arms. The SO-101's printed parts are yellow, and its motors and the drawer handle are black. To choose a palette, 23 candidate colours were rendered as the cube on A0's own frames (`python -m sim.colour_contrast data/a0`, report in `check/colours.txt`; one episode per variant, every 20th frame, each episode's lighting, overview camera). Each was measured as the RGB distance (0–1 per channel) between the cube's mean rendered colour and each surround, split by segmentation: the arms' yellow parts, their black motors, the table, drawer, handle and cabinet. A colour's score is its weakest surround (median over frames). Averaging over the whole arm, as the first check did, hid black: it differs a lot from the yellow parts but matches the motors.

| Colour | Weakest surround | Contrast | Use |
|---|---|---|---|
| teal | handle | 0.74 | train |
| violet | drawer | 0.68 | train |
| indigo | handle | 0.68 | — |
| magenta | table | 0.66 | eval |
| crimson | table | 0.66 | train |
| green | arm yellow | 0.60 | train (kept) |
| cyan | drawer | 0.55 | eval (kept) |
| blue | drawer | 0.54 | train (kept) |
| maroon | handle | 0.52 | eval |
| red | arm yellow | 0.48 | train (kept) |
| navy | handle | 0.45 | eval |
| purple | drawer | 0.42 | dropped (train) |
| pink | drawer | 0.28 | dropped (eval) |
| lime | arm yellow | 0.26 | dropped (eval) |
| yellow | arm yellow | 0.19 | dropped (train) |
| orange | arm yellow | 0.19 | dropped (train) |
| black | handle (motors 0.13) | 0.08 | dropped (eval) |

Also measured and rejected: brown 0.44, dark green 0.40, grey 0.26, sky 0.21, mint 0.16, white 0.10 (against the cabinet).

- **Choice:** every colour scores ≥ 0.45. Train: red, green, blue, violet, teal, crimson. Eval: magenta, cyan, navy, maroon. Each eval colour lies between training colours: its nearest training colour is 0.29–0.37 away in nominal RGB (navy–violet is the closest pair; in the old palette the closest was 0.31). So the eval-colours condition still tests unseen colours, as before, not a whole unseen hue range. Splitting by hue family (blues and greens to train, reds to eval) would widen the gap to 0.54, but it would test a harder, different question.
- **Same starts:** both palettes keep 6 and 4 colours, so each seed draws the same index and the same start, distractors and lighting. Only the colour changes.
- **Reference:** held-out object colours as an evaluation shift follow The Colosseum (Pumacay et al. 2024), where object colour is one of the perturbation factors. Choosing the palette by measured contrast in the rendered scene, rather than by name, is our own step: the SO-101's yellow parts and black motors make the usual primary colours a poor fit.
- **Check:** `check_data` reports each episode's weakest-surround contrast and flags it below `MIN_CONTRAST` = 0.4. On one episode per variant with the new palette (20 episodes): 0 flags, all successful. The lowest contrast is blue against the drawer, 0.42 (its median over the colour sweep was 0.54); the per-colour medians are red 0.47, green 0.59, violet 0.64, crimson 0.64, teal 0.80. The eval colours were measured in the sweep only (they are recorded with `split="eval"` at evaluation).
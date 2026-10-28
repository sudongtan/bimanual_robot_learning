
## Next step: recording

Nothing writes episodes to disk yet. The plan: record a small batch, inspect it, fix the expert or recorder, then run a training pilot (e.g. 25 episodes per task and arm, a short ACT run, evaluation in sim).

**Hook.** Every `run_*` takes `rec`, called as `rec(sim)` after every 2 ms physics step during the task. The expert runs at 50 Hz (`CONTROL_EVERY` = 10 steps, zero-order hold, design point 1): on the call after the first step of each 20 ms interval `sim.ticked` is true, and `sim.tick_state` (the physics state before the interval, `STATE_SPEC`), `sim.applied_ctrl` (the command held over it, with demonstration noise), `sim.nominal_ctrl` (the same without noise) and `sim.subtask` are one frame. Settling steps before the task starts are not passed to `rec`. The 2 s hold at the end (`HOLD_S`, in `finish`: both arms at home, nothing moving) is, so every episode demonstrates staying still after the task (the stop tests in `1tasks_scene.md`). Episode lengths including the hold (one episode each, 50 fps frames): S1 10.5 s (524), S2 9.9 s (494), S3 12.9 s (644), S4 11.7 s (586), T1 20.4 s (1,020), T2 21.1 s (1,056).

**Proposed recorder:**
- Sample every 10th physics step (50 fps, as ACT's sim data; see the frame-rate note below).
- Per frame: `qpos` (enough to re-render any camera later; the full physics state would also need `qvel`, actuator state and `ctrl`, see `mj_getState`); `observation.state` = the 12 actuated joint positions; `action` and `action_nominal` (see design points); optionally the three camera images.
- Per episode: task, command, seed, success, and `scene` (a dict; store as JSON). `scene` must hold the drawn values of the render-only factors (colours, lighting), not just the seed, so a re-render reproduces the episode.
- LeRobot feature names: `observation.state`, `action`, `observation.images.{overview,left_wrist,right_wrist}`, with the 12 joint names in the feature metadata, so ACT, SmolVLA and π0 load it directly.
- Keep failed episodes but flag them; train only on successful ones.
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

**Checks on the first batch:**
1. **Replay:** play the recorded actions in a fresh sim from the recorded start state (each held for 10 steps) and confirm every recorded state matches exactly (as `test_replay_is_exact`) and the task still succeeds.
2. **Images:** the cube is visible and not hidden by the arms; the wrist cameras see the cube, handle and drawer inside when it matters; colours and lighting look right.
3. **Actions:** no jumps (watch the carry to the far-side S4/T1 targets), no long idle stretches, sensible gripper timing.
4. **Labels:** the command matches what happens and the named arm is the one moving.

**Data notes:**
- T1 episodes are also S1 + S4 demos, T2 episodes S3 + S2: mark the boundary and label each part with its sub-command (design point 5).
- Balance episodes per command, both directions of each composite, and both start kinds of S2 (design point 6).

**Open:**
- Episode counts. Pilot: 25 per task and arm. Default for the full set: 50 per command (ACT used 50 sim demos per task; LIBERO has 50 per task, the set SmolVLA and π0 are fine-tuned on).
- An automated render check (cube pixels and contrast at 480 × 640 and at 224 over many seeds). Match the cube by its segmentation ID, not its colour: lighting changes the rendered colour too much for a colour match.
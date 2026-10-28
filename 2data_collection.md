
## Next step: recording

Nothing writes episodes to disk yet. The plan: record a small batch, inspect it, fix the expert or recorder, then run a training pilot (e.g. 25 episodes per task and arm, a short ACT run, evaluation in sim).

**Hook.** Every `run_*` takes `rec`, called as `rec(sim)` after every 2 ms physics step during the task. Settling steps before the task starts and the 0.5 s settle at the end (in `finish`) are not passed to `rec`. Episode lengths (one episode each): S1 8.4 s, S2 7.8 s, S3 9.6 s, S4 9.6 s, T1 18.5 s, T2 18.2 s.

**Proposed recorder:**
- Sample every 10th physics step (50 fps, as ACT's sim data; see the frame-rate note below).
- Per frame: `qpos` (enough to re-render any camera later; the full physics state would also need `qvel`, actuator state and `ctrl`, see `mj_getState`); `observation.state` = the 12 actuated joint positions; `action` and `action_nominal` (see design points); optionally the three camera images.
- Per episode: task, command, seed, success, and `scene` (a dict; store as JSON). `scene` must hold the drawn values of the render-only factors (colours, lighting), not just the seed, so a re-render reproduces the episode.
- LeRobot feature names: `observation.state`, `action`, `observation.images.{overview,left_wrist,right_wrist}`, with the 12 joint names in the feature metadata, so ACT, SmolVLA and π0 load it directly.
- Keep failed episodes but flag them; train only on successful ones.
- Separate seed ranges for training and evaluation, and use `Randomization(split="eval")` for evaluation episodes.
- One LeRobot dataset for all tasks, with the command as each episode's task (fits the VLA plan; ACT can train on a filtered subset).

**Design points:**

1. **Action timing: hold each command for one frame.** `rec(sim)` runs after `mj_step`, so `sim.applied_ctrl` at a frame is the command that produced that frame's state, not the next one. The expert also changes its command every 2 ms, so a 50 fps sample skips the 9 commands in between, and replay can drift (most likely on the carry to the far-side S4/T2 targets). ACT's scripted sim policy runs at the control rate, so each recorded action is held for the whole 20 ms and replay is exact.
   - Decision: add a recording mode where the expert updates `ctrl` only every 10th step (50 Hz, zero-order hold). Record pairs of (observation before the 10 steps, command held during them). The replay check below then has to be exact.
   - Fallback if the expert can't run at 50 Hz: `action[k]` = the command at frame k+1 (the target reached by the end of the interval).
2. **Noisy and clean action labels: store both.** `action` = the command sent, including demonstration noise (as ACT records it). `action_nominal` = the expert's command without noise. DART (Laskey et al. 2017) runs the noisy commands but trains on the clean ones: they are position targets, so they point back to the planned path and act as recovery labels. Pick which to train on by comparing both in the pilot.
3. **Frame rate: 50 fps.** Matches ACT's sim data and π0, and divides the 2 ms timestep evenly (one command per 10 steps). Training and evaluation are both in sim, so there is no real-robot rate to match.
4. **Image resolution.** At 224 × 224 the cube is only ~12 px in the overview. ACT rendered at 480 × 640, and LeRobot stores video and resizes in the policy. Record at 480 × 640 (or re-render later from `qpos` + `scene`) and resize per policy.
5. **T2 sub-commands as per-frame labels.** A LeRobot episode has one `task` string, so keep the full T2 command as the task and add a per-frame `subtask_index` (0 = S1 part, 1 = S4 part) plus the frame where the drawer is open. Don't cut T2 into two episodes: the S4 half wouldn't start from home.
6. **Balance per command, not just per task × arm.** S1, S2, S3 and T1 have 2 commands each (arm, or hand-off direction); S4 and T2 have 6 (arm × target). Equal counts per task × arm leave each S4 command with a third of the episodes of an S1 command. Default: equal episodes per command, plus both start kinds of S1–S3.

**Checks on the first batch:**
1. **Replay:** play the recorded actions in a fresh sim from the same start (each held for one frame) and confirm the task still succeeds and the joint trajectory matches the recording.
2. **Images:** the cube is visible and not hidden by the arms; the wrist cameras see the cube, handle and bar when it matters; colours and lighting look right.
3. **Actions:** no jumps (watch the carry to the far-side S4/T2 targets), no long idle stretches, sensible gripper timing.
4. **Labels:** the command matches what happens and the named arm is the one moving.

**Data notes:**
- T2 episodes are also S1 + S4 demos: mark the frame where the drawer is open and label each part with its sub-command (design point 5).
- Balance episodes per command, both hand-off directions, and both start kinds of S1–S3 (design point 6).

**Open:**
- Episode counts. Pilot: 25 per task and arm. Default for the full set: 50 per command (ACT used 50 sim demos per task; LIBERO has 50 per task, the set SmolVLA and π0 are fine-tuned on).
- An automated render check (cube pixels and contrast at 224 × 224 over many seeds).
- T1 left/right targets (optional): would need the bar put down at the ±8 cm spots and validated.
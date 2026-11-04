## Training and evaluation

Results per plan step ([1tasks_scene.md](1tasks_scene.md), "Minimal plan"), and the bugs and findings that changed how policies are trained or evaluated. Recording design and data checks are in [2data_collection.md](2data_collection.md).

**How a number is made.**
- Training: `lerobot-train` (LeRobot 0.6.1), ACT at LeRobot's setting (100k steps, batch 8), on this Mac's GPU (`mps`) at 240 × 320 (2data_collection.md, design point 4), ~2.4 steps/s, ~12 h per run. Scripts: `scripts/train_step1.sh`, `scripts/step2_3.sh`; for a rented GPU, `cloud/train.sh` ([4cloud_training.md](4cloud_training.md)).
- Evaluation: `python -m sim.evaluate CKPT --variants S1:A --episodes 50 --temporal-ensemble 0.01`. Rollouts start from `EVAL_SEEDS` (the same for every variant and model, never trained on), run a fixed horizon (the task's longest demo plus 2 s, as CALVIN) and are scored by the expert's own success checks (`sim/episode.py`), plus the stop rate (S1, S3) and continue rate (T1, T2). Each rollout logs its checks, the cube's slide in the drawer (`cube_slide_mm`, meaningful for S1), and its final position with the nearest named spot (`cube_final`, `nearest_target`) to `episodes.jsonl`.

## Results

### Plan step 1: ACT on S1 arm A (50 episodes)

Bar: ≥ 80% on the train condition.

| Checkpoint | Run | Rollouts | Success | Stopped | Failed checks | Cube slide (median) | Output |
|---|---|---|---|---|---|---|---|
| 40k, no floor | chunks open loop | 10 | 0/10 | 0/10 | drawer not opened (eval starts; 2/10 on training starts) | — | `outputs/s1a_act_action_nofloor` |
| 20k | chunks open loop | 10 | 1/10 | 1/10 | drawer_open 8, cube_kept_place 7, named_arm_acted 1 | — | `eval_020000` |
| 40k | chunks open loop | 10 | 2/10 | 2/10 | cube_kept_place 6, drawer_open 2, named_arm_acted 2 | — | `eval_040000` |
| 60k | chunks open loop | 10 | 4/10 | 4/10 | cube_kept_place 5, drawer_open 1, named_arm_acted 1 | 5.1 mm | `eval_060000` |
| 60k | temporal ensembling 0.01 | 10 | 10/10 | 10/10 | — | 4.2 mm | `eval_060000_te` |
| **100k** | **temporal ensembling 0.01** | **50** | **44/50 (88%)** | **44/50** | **cube_kept_place 6** | **4.2 mm** | `eval_100000_te` |

All on S1:A eval starts, train condition, trained on `action`; outputs under `outputs/s1a_act_action/` unless given.

**Result: passes (88%).** Every one of the 50 rollouts opened the drawer, and every success stopped. All 6 failures are the cube sliding in the drawer past S1's 5 mm limit: 5.1–7.7 mm, four of them ≤ 5.4 mm. Cube slide over all 50: median 4.2 mm, 10th–90th percentile 3.1–5.2 mm. The expert's slide over the 50 recorded episodes is 0.9–4.0 mm (median 2.2; `cube_moved` in `data/s1a/sim/episodes.jsonl`): the limit is above every demonstration, and the policy pulls about twice as hard as the expert. The remaining gap is the pull, not grasping or stopping.

**Clean labels (`action_nominal`): no difference; keep `action`.** The second half of step 1 (2data_collection.md, design point 2: DART trains on the noise-free command while the noisy one is run). Same training (100k steps; `outputs/s1a_act_nominal`, data `data/s1a_nominal`) and the same 50 eval starts with temporal ensembling:

| Label | Success | Stopped | Failed checks | Cube slide median (10th–90th pct, max) |
|---|---|---|---|---|
| `action` (noisy, as run) | 44/50 (88%) | 44/50 | cube_kept_place 6 | 4.2 mm (3.1–5.2, 7.7) |
| `action_nominal` (clean) | 41/50 (82%) | 41/50 | cube_kept_place 9 | 4.5 mm (3.7–5.2, 6.8) |

Paired by start: 38 succeed under both, 6 only with `action`, 3 only with `action_nominal`. Six against three is well within chance (McNemar's exact test, p ≈ 0.5), and per start the clean labels slide the cube 0.2 mm more (median). Both fail the same way, only by the cube sliding, mostly just over 5 mm. So at this demonstration noise the label choice doesn't matter for S1; keep `action`, the command the actuators actually got (as ACT records its sim data), and don't make `--nominal` views for later steps. DART's case for clean labels is recovery from compounding error; S1 shows none to recover from (every rollout opens the drawer), so a longer task (T1) is where it could still show up.

**Held-out colours (`--condition eval_colours`, 2026-11-02): 44/50 (88%), the same as the train colours**, all 6 failures the cube sliding past 5 mm (`eval_100000_te/eval_colours`). The arms held still in all 50 rollouts (`held_still`, logged from this run on: the last 2 s alone, whatever the outcome), so the stop behaviour itself is 50/50 and the stop rate is below it only through the slide check.

### Minimal plan steps 2-3: one ACT with the one-hot variant input, 6 variants (300 episodes)

Recorded `data/min` (S1 A/B, T1 A/B × left/right, 50 each; 0 expert failures; checks: 11/300 flags, all red cubes under dim light at 0.34-0.40 contrast with the table, as in `data/s1a`). View `data/min_onehot` (`--variant-input`, no reweighting: finding 5). ACT 100k steps (`outputs/min_act`, `scripts/step2_3.sh`; finished 2026-10-31 22:22, evaluated by 2026-11-01 04:12). The dataset is 10× step 1's, so 100k steps at batch 8 is ~3 epochs (step 1: ~30).

Interim, 40k checkpoint (~1.3 epochs), 10 rollouts per variant, temporal ensembling (`eval_040000_te`):

| Variant | Success | Stop (S1) / continue (T1) | Failed checks |
|---|---|---|---|
| S1:A | 7/10 | stop 7/10 | cube_kept_place 3 (5.2-5.7 mm) |
| S1:B | 10/10 | stop 10/10 | — |
| T1:A:left | 6/10 | continue 10/10 | cube_at_target 4 |
| T1:A:right | 6/10 | continue 10/10 | cube_at_target 4, first_arm_kept_to_its_step 1 |
| T1:B:left | 4/10 | continue 10/10 | cube_at_target 6 |
| T1:B:right | 8/10 | continue 10/10 | cube_at_target 2 |

The task input works already: every S1 success stops and every T1 rollout continues to the hand-over (from the same start scene). T1's failures are all the cube not ending within 2 cm of its spot; from the logged slide some were never carried out (9-50 mm from the drawer start), others ended about as far away as a placed cube, which may be the wrong side or a near miss. From this checkpoint on, `sim/evaluate.py` also logs the cube's final position and the nearest named spot (`cube_final`, `nearest_target`) to tell these apart.

**Final, 100k checkpoint (~3 epochs), 50 rollouts per variant, temporal ensembling (`eval_100000_te`):**

| Variant | Success | Stop (S1) / continue (T1) | Failed checks |
|---|---|---|---|
| S1:A | 30/50 (60%) | stop 30/50 | cube_kept_place 20 (5.0-6.1 mm) |
| S1:B | 42/50 (84%) | stop 42/50 | cube_kept_place 8 (5.0-6.3 mm) |
| T1:A:left | 49/50 (98%) | continue 50/50 | cube_at_target 1 |
| T1:A:right | 49/50 (98%) | continue 50/50 | cube_at_target 1 |
| T1:B:left | 47/50 (94%) | continue 50/50 | cube_at_target 3 |
| T1:B:right | 50/50 (100%) | continue 50/50 | — |

- **The command picks the task and the target.** From one start scene, no S1 rollout took the cube out (every S1 slide ≤ 6.3 mm: the cube stayed in the drawer), every T1 rollout continued to the hand-over (200/200), and no T1 cube ended at the other target (0/200 wrong side; `nearest_target`). So the imbalance (finding 5) produced no lean either way, and no reweighting is needed.
- **T1: 195/200 (97.5%).** The 5 failures: 3 near misses for T1:B:left (cube 0.5-2.8 cm from its spot, or not upright), and one start (seed 100000) where both arm-A commands left the cube in the drawer after the second arm's grasp missed (video `T1_A_left_100000_fail.mp4`).
- **S1: 72/100, every failure the 5 mm slide limit.** All S1 failures slid the cube 5.0-6.3 mm. S1:A is below the single-task policy of step 1 (88%), with a harder pull (median slide 4.9 mm vs 4.2; S1:B 4.3). Possibly because S1's frames are seen ~3 times here against ~30 in step 1; not tested, and not worth a longer run (finding 9: the gap is the pull, not the task); the stop rate equals success because it requires it, so it doesn't separate "held still" from the slide.

**Held-out colours, same 100k checkpoint (`--condition eval_colours`, 50 per variant, `eval_100000_te/eval_colours`, finished 2026-11-02 17:58):**

| Variant | Held-out colours | Train colours | Failed checks (held-out) |
|---|---|---|---|
| S1:A | 32/50 (64%) | 30/50 | cube_kept_place 18 (5.1-6.5 mm); held still 50/50 |
| S1:B | 45/50 (90%) | 42/50 | cube_kept_place 5 (5.0-6.2 mm); held still 50/50 |
| T1:A:left | 49/50 | 49/50 | cube_at_target 1 (seed 100000, cube left in the drawer, as with train colours) |
| T1:A:right | 49/50 | 49/50 | cube_at_target 1 (seed 100000, the same) |
| T1:B:left | 46/50 | 47/50 | cube_at_target 4, all near misses at the left spot |
| T1:B:right | 50/50 | 50/50 | — |

**No colour gap.** Held-out T1 194/200 against 195/200; every T1 rollout continued and none ended on the wrong side; S1 held still 100/100 and failed only by the slide. T1 is the part that tests colour, since the second arm has to find and grasp the cube; S1 needs only the handle. As in step 1 (88% both ways), unseen cube colours don't hurt ACT here.


### Minimal plan step 4: SmolVLA with the text commands (300 episodes)

`lerobot/smolvla_base` fine-tuned with LeRobot's recipe on `data/min` (20k steps at batch 64 on a rented RTX 4090, ~5 h; vision encoder and language model frozen, only the action expert trained: `train_expert_only`, LeRobot's default). Evaluated as it runs by default: 50-action chunks executed open loop, 10 flow-matching steps (`scripts/eval_smolvla.sh`, `outputs/min_smolvla/eval`, 50 rollouts per variant, ~60 s per rollout on the Mac's GPU). T1 misses: distance from the cube to the commanded spot, and whether it ended nearest that spot.

| Variant | Train | Held-out colours | Held-out wording | Step 3 ACT (train) |
|---|---|---|---|---|
| S1:A | 40/50, stop 39 (cube_kept_place 9, 5.1-8.7 mm; other arm moved 1) | 34/50 (cube_kept_place 11; other arm moved 5; drawer 1) | **8/50** (drawer_open 28, other arm moved 16, cube_kept_place 12) | 30/50 |
| S1:B | 36/50, stop 36 (cube_kept_place 14, 5.1-8.7 mm) | 37/50 (cube_kept_place 13) | 41/50, stop 33 (cube_kept_place 8; other arm 1) | 42/50 |
| T1:A:left | 30/50, continue 49 (miss median 2.8 cm, 19/20 nearest the right spot) | 31/50, continue 50 (2.5 cm, 14/19) | **0/50**, continue 0 (second arm never acted; opener idle 19) | 49/50 |
| T1:A:right | 27/50, continue 49 (3.5 cm, 14/23) | 27/50, continue 49 (4.9 cm, 11/23) | **0/50**, continue 0 (opener idle 16) | 49/50 |
| T1:B:left | 32/50, continue 50 (6.7 cm, 8/18) | 28/50, continue 50 (12.4 cm, 8/22) | **0/50**, continue 0 (neither arm acted, 50/50) | 47/50 |
| T1:B:right | 22/50, continue 50 (4.2 cm, 22/28) | 36/50, continue 50 (3.7 cm, 13/14) | **0/50**, continue 0 (neither arm acted, 50/50) | 50/50 |

**Train: level with ACT on S1, well below it on T1.** S1 76/100 against ACT's 72/100, failing as ACT does, by the slide (finding 9). T1 111/200 (56%) against 195/200: SmolVLA hands over to the second arm (198/200), so it reads the task, but the placement is loose, every failure is cube_at_target. With arm A placing (T1:B) misses are larger and often not at the commanded spot (T1:B:left: median 6.7 cm, 8/18 nearest it); with arm B placing (T1:A) they are mostly near misses (T1:A:left: median 2.8 cm, 19/20 on the commanded spot). Not separated yet: the policy itself against the way it is run (50 actions open loop, ~1.7 s, where ACT re-plans every frame; finding 2 showed how much that matters for ACT).

**Held-out colours: no gap** (S1 71/100, T1 122/200 against 76 and 111), as for ACT.

**Held-out wording: the policy loses the task.** The held-out phrasing is a new sentence shape for each task ("arm A, pull the drawer open"; "arm A opens the drawer and arm B moves the cube to the left of the table", without the "then" and "take the cube out" both training phrasings have). Under it T1 fails 200/200: in T1:A arm A reaches into the drawer area but the second arm never moves; in T1:B neither arm does its step (arm A drifts, arm B stays home). S1:A drops to 8/50 (drawer not opened 28, the other arm moves 16); S1:B is unaffected (41/50). With only two training phrasings per task and the language model frozen, the action expert learned the language model's output for those two sentences; a third shape is out of distribution. This is the opposite of LIBERO-PRO and LIBERO-Plus (2025) on LIBERO, where each scene points to its task: OpenVLA, π0 and π0.5 kept 82–98% success under synonym rewording (LIBERO-PRO), OpenVLA-OFT kept its success with an empty instruction on LIBERO-Object (LIBERO-Plus), and LIBERO-PRO reports the same actions for nonsense instructions; both read this as the models ignoring the language. Here S1 and T1 share the scene, so the text can't be ignored (an empty command changes the behaviour), and rewording breaks it instead. What the two agree on: neither shows the instruction being read for its meaning (LIBERO-PRO's task-level changes and LIBERO-Plus's swapped target objects drop success to near zero). ACT and DP are not exposed to this (they get the variant as a fixed vector), which is the trade the one-hot input makes. Step 5's `stress_wording` (eight more phrasings) and `stress_blank_command` show whether any rewording breaks it or only some shapes.

### Minimal plan step 5: stress tests, ACT, SmolVLA and DP

One factor past its training range at a time (1tasks_scene.md, step 5), on S1:A and T1:A:left, the first 20 eval starts (`scripts/stress.sh`, 2026-11-03; ACT with temporal ensembling, outputs in each policy's eval directory under `stress_*`). "Same starts" is the train condition on those 20 starts. Successes out of 20, failed checks in brackets.

| Condition | ACT S1:A | ACT T1:A:left | SmolVLA S1:A | SmolVLA T1:A:left |
|---|---|---|---|---|
| Same starts, train condition | 13 | 19 | 18 | 12 |
| `stress_distractors` (all 3) | 14 (slide 6) | 19 (target 1) | 15 (slide 4, other arm 1) | 14 (target 6) |
| `stress_start_pose` (3×) | 6 (**other arm 10**, slide 4) | 17 (target 3) | 5 (**other arm 14**, slide 4, drawer 2) | 14 (target 6; continue 19) |
| `stress_cube_yaw` (±40°) | 13 (slide 7) | 18 (target 2) | 12 (slide 8, other arm 1) | **9** (target 11; continue 19) |
| `stress_wording` (8 more phrasings) | — | — | 10 (drawer 9, other arm 5) | **0** (target 20; continue 8) |
| `stress_blank_command` (empty) | — | — | **0** (drawer 20, named arm 20, other arm 20) | — |

DP (10 DDIM steps, `scripts/eval_dp.sh`, `outputs/min_dp/eval_100000_ddim10`), same format; same starts S1:A 17, T1:A:left 17:

| Condition | DP S1:A | DP T1:A:left |
|---|---|---|
| `stress_distractors` | 16 (slide 4) | **10** (target 10; continue 20) |
| `stress_start_pose` | 8 (other arm 8, slide 7) | **12** (target 8; continue 20) |
| `stress_cube_yaw` | 19 (slide 1) | **10** (target 10; continue 20) |

**Start pose, rerun with the corrected idle-arm measure (2026-11-03, `scripts/rerun_start_pose.sh`; finding 11).** Rollouts now log `idle_arm_rad.away`, how far the idle arm's joints go outside the range between their start and their usual pose. With the arm counted as left only above 0.1 rad on that measure, S1:A under `stress_start_pose`: ACT 16/20 (task 18/20), SmolVLA 16/20 (task 19/20), DP 13/20 (task 19/20), against 6, 5 and 8 with the old check. ACT's and SmolVLA's idle arm never went more than 0.08 rad outside its range; one DP rollout moved it 0.93 rad, a real movement. T1:A:left in the rerun: ACT 17/20, SmolVLA 14/20 (as in the first run). ACT and SmolVLA reruns in `outputs/min_act/eval_100000_te_rerun` and `outputs/min_smolvla/eval_rerun`; DP's own run already logged the measure.

**S1 on the task** (drawer opened, arms held still; start pose with the corrected measure), same starts → distractors / turned cube / start pose: ACT 20 → 20 / 20 / 18, SmolVLA 20 → 19 / 19 / 19, DP 20 → 20 / 20 / 19. None of the three changes affects S1 for any policy; the drops in the success rates above are the slide limit (finding 9) and, for start pose, the check (finding 11).

**T1:A:left** under the changes: ACT 19 → 19 / 18 / 17, SmolVLA 12 → 14 / 9 / 14, DP 17 → **10 / 10 / 12**. DP's T1 drops by 5–7 under every change, more than the 3–4 that 20 rollouts can't resolve; every DP failure is the cube away from its spot, and it hands over 20/20 each time. Not from DDIM: on the first 10 of these starts, without a change, DP's T1 is 8/10 with DDIM and 10/10 with the 100 DDPM steps (step 3b). SmolVLA's T1 can't be compared this way, since it is at 12/20 without any change.

**Distractors: no effect** on either policy. **Turned cube: ACT unaffected; SmolVLA's T1 drops** (9 against 12; all misses at the target), the clearest physical robustness gap, and against the VLA. 20 rollouts per cell: differences of 3–4 are within noise. **Start pose: the S1 drop is mostly the check** (finding 11): under 3× start noise the idle arm often starts 0.08–0.15 rad from its usual pose, both policies move it part of the way back, and `other_arm_stayed_home` (a joint > 0.1 rad from its start) fails. Whether that happens follows the idle arm's start offset in 15 of 20 starts for each policy (no failure below 0.04 rad, most above 0.12). Without that check S1:A would be about 16 (ACT) and 15 (SmolVLA), as on the same starts; T1 is barely touched (17, 14).

**Language.** More phrasings: S1:A 10/20, T1:A:left 0/20. Phrasings close to the training ones work ("with arm A, open the drawer" 6/6, "please open the drawer with arm A" 1/1); other structures fail ("can you open the drawer using arm A?" 0/3, "arm A: open drawer" 0/1). In T1, "with arm A, open the drawer, then with arm B, …" and "first …; then arm B …" hand over but miss the target; "arm A: open drawer. arm B: …", "get arm A to … and arm B to …" never start arm B. Empty command: 20/20 the same behaviour, arm B moves, arm A doesn't, the drawer stays shut: the text drives the policy (it doesn't fall back on a scene default such as S1), but it is matched by form, not read, as `eval_wording` showed. Per-phrasing counts are small.

### Minimal plan step 3b: Diffusion Policy with the one-hot input (training, checks at 5k, 20k, 100k)

Training on the rented 4090 (`outputs/min_dp`, 100k steps at batch 64, 0.33 s per step; ended 2026-11-03 02:52 CET. The 50 GB disk filled at the 55k save (3.3 GB per checkpoint with the optimizer state); training resumed from the 50k checkpoint with LeRobot's `--resume`, which restores the optimizer, scheduler, RNG and data order, so steps 50k-55k were repeated, not skipped; `cloud/dp_resume.sh`). Early checks with `scripts/check_dp.sh` (finding 10): offline replay of 6 training episodes and 3 rollouts each of S1:A and T1:A:left, on the CPU (~2.5 min per S1 rollout, ~4.5 min per T1).

| Checkpoint | Passes over `data/min` | Replay error, mean (worst joint) | S1:A | T1:A:left |
|---|---|---|---|---|
| 5k | ~1.3 | 0.018-0.033 rad (0.04-0.07, the wrist) | 0/3: drawer not opened, the other arm left home | 0/3: cube not at target |
| 20k | ~5 | 0.010-0.016 rad (0.02-0.03) | 0/3: drawer opened, but **the other arm left home** every time (stop 0/3) | 0/3: 2 carried the cube to the left side but 3.5 cm past the spot (limit 2 cm); 1 never started the second arm |
| 100k | ~25 | 0.0018-0.0034 rad (0.004-0.007) | 4/5; the other arm held still in 5/5; the fail slid the cube 5.2 mm (cube_kept_place; the 4 successes 3.8-4.0 mm) | 5/5, continue 5/5 |

DP is learning (the replay error halved, the drawer now opens, the cube goes to the commanded side). The lead to check at 100k: under the S1 command DP goes on as in T1, which step 3's ACT never did (0/100). A possible reason is how the command enters: ACT gives the one-hot vector its own encoder token, DP appends its 11 numbers to the image and state features in the U-Net's conditioning. Three rollouts are a lead, not a finding. **At 100k the lead is gone:** the other arm stays home under the S1 command (5/5), so at 20k DP had not yet learnt to stop, and the command input works. The replay error at 100k is below ACT's (0.005-0.007). Five rollouts per variant (`outputs/min_dp/eval_100000`, CPU, ~3 min per S1 and ~5 min per T1 rollout) are a check, not a result.

**Full evaluation with 10 DDIM steps (`scripts/eval_dp.sh`, `scripts/eval_dp_t1.sh`; 2026-11-03 14:32 to 2026-11-04 05:18).** DP denoises with the 100 DDPM steps it was trained with, ~3 min per S1 and ~5 min per T1 rollout, so 600 rollouts would take more than a day. `sim/evaluate.py --ddim 10` runs the same model with 10 DDIM steps on the same noise schedule, as Chi et al. (2023) run their real-robot policies. Offline replay, episodes 0 and 179: 0.0023 and 0.0034 rad mean error with DDIM, 0.0022 and 0.0028 with DDPM. The evaluation is the same as ACT's and SmolVLA's (train and held-out colours, 6 variants × 50; the three stress conditions, S1:A and T1:A:left × 20), plus a check: the first 10 starts of S1:A and T1:A:left with the 100 DDPM steps, against the same starts with DDIM.

`eval_dp.sh` ran the train and held-out-colours conditions without `--variants`, so `sim.evaluate` took its default, all 20 variants: it evaluated S1:A and S1:B, went on to S2:A (not in the minimal plan) and stopped there on an error in tallying S2's checks, before any T1 (~3.3 h of the Mac). The S1 results are kept (`outputs/min_dp/eval_100000_ddim10`); T1 was rerun with the variants listed (`scripts/eval_dp_t1.sh`, `outputs/min_dp/eval_100000_ddim10_t1`). The S2 tally error is in `sim/evaluate.py`'s summary and affects no minimal-plan run.

**DDIM check.** On the same starts, without a change: S1:A 9/10 with DDPM, 9/10 with DDIM; T1:A:left 10/10 with DDPM (median distance to the spot 1.25 cm), 8/10 with DDIM (median 1.35 cm; the two misses 2.3 and 3.0 cm). Two starts differ, within noise; DDIM is used for every DP result below. `outputs/min_dp/eval_100000_ddpm`.

**Final, 100k checkpoint, 50 rollouts per variant, 10 DDIM steps:**

| Variant | Train | Held-out colours | Failed checks (train; colours) | Step 3 ACT (train) |
|---|---|---|---|---|
| S1:A | 36/50, stop 36 | 37/50 | cube_kept_place 14; 13 | 30/50 |
| S1:B | 46/50, stop 46 | 44/50 | cube_kept_place 4; 6 | 42/50 |
| T1:A:left | 35/50, continue 50 | 30/50, continue 50 | cube_at_target 15 (miss median 2.7 cm, all nearest the named spot); 20 (3.4 cm, 19/20) | 49/50 |
| T1:A:right | 46/50, continue 50 | 41/50, continue 50 | cube_at_target 4 (3.1 cm, 4/4); 9 (2.5 cm, 7/9) | 49/50 |
| T1:B:left | 50/50, continue 50 | 46/50, continue 50 | —; cube_at_target 4 (2.4 cm) | 47/50 |
| T1:B:right | 50/50, continue 50 | 48/50, continue 50 | —; cube_at_target 2 (2.2 cm) | 50/50 |

S1 on the task (`sim.s1_report`): 100/100 both conditions, slide median 4.1–4.5 mm (train), success 47–50/50 at 6 mm and 50/50 at 8 mm, as ACT. **Train: S1 82/100, T1 181/200** (ACT 72 and 195, SmolVLA 76 and 111); **held-out colours: S1 81/100, T1 165/200**. DP follows the command every time (no S1 rollout went on, every T1 rollout handed over), and its T1 misses are placement, mostly just past the 2 cm limit, when arm B places (T1:A); with arm A placing (T1:B) it succeeds in all 100 training-condition rollouts. Held-out colours cost DP's T1 16 rollouts (181 → 165), where ACT and SmolVLA showed no gap; mostly T1:A:left (35 → 30) and T1:A:right (46 → 41).

### Planned: ACT and DP reading the text commands (not run)

**Question.** SmolVLA loses the task when a command is worded differently (README Q2). Two explanations fit: something about the VLA, or simply training on two phrasings per task, which any policy reading text would suffer from. ACT and DP can't tell these apart, because they get a code, not words.

**Experiment.** ACT and DP that read the same commands as SmolVLA, through a frozen sentence embedding, the usual way imitation policies are compared with a VLA: OpenVLA's Diffusion Policy baseline on LIBERO conditions on DistilBERT embeddings of the task label; LIBERO's own baselines use BERT embeddings, as a token next to the visual tokens.

- **Input:** each episode's command, encoded by frozen DistilBERT (as OpenVLA's baseline), in place of the one-hot code and in the same input slot (`observation.environment_state`): ACT takes it as its own encoder token, DP adds it to its conditioning. Only the code changes to a sentence embedding; data, cameras and training are as in the one-hot runs.
- **Data and training:** `data/min` with its recorded commands (two phrasings per task), as SmolVLA; ACT 100k steps at batch 8 on the Mac, DP 100k steps at batch 64 on a rented GPU, as before.
- **Evaluation:** the same as SmolVLA in steps 4-5: training phrasings and held-out sentence shape (6 variants × 50), eight more phrasings and an empty command (S1:A and T1:A:left × 20), and the same task measures.

**What each comparison would show.**

| Comparison | Reads | If similar | If different |
|---|---|---|---|
| Text ACT/DP against one-hot ACT/DP, training phrasings | Whether reading text costs anything when the wording is known | Reading text is free on known wording (LIBERO found task-ID and instruction embeddings perform about the same) | Text conditioning itself costs something here |
| Text ACT/DP against SmolVLA, new wording | Whether the wording failure is the VLA's | Training on two phrasings per task causes it, whatever the policy | It is specific to SmolVLA |

**What it can't show.** If the failure is specific to SmolVLA, whether that comes from how it combines text with images or from its pretraining: these differ together between a sentence-embedding ACT and a VLA (separating them needs SmolVLA fine-tuned from different starting weights). Nor does it say anything about the placement gap (README Q3).

**Needs:** a dataset view that adds the command embedding (as `sim.dataset_view --variant-input` adds the code), the same in `sim.evaluate` for each rollout's command, then the two trainings and the evaluations.

## Findings and bugs

**1. Normalisation floor (bug, fixed).** LeRobot normalises each joint by its standard deviation over the dataset. In S1 arm A the idle arm's gripper never moves (std 7 × 10⁻⁹), so physics jitter of ~3 × 10⁻⁷ rad in a rollout became normalised inputs of 15–50 (all others within ±3), and the network could tell training episodes apart by that jitter. ACT fitted the recorded actions to 0.003–0.009 rad offline but opened the drawer in 0/10 eval starts and 2/10 training starts at 40k steps (`outputs/s1a_act_action_nofloor`). Ruled out first: the rollout loop (same first observation and physics state as the recording), image compression (rendered and video-decoded images give equally accurate predictions), and temporal ensembling on its own (1/8).
- Fix: `STD_FLOOR` = 0.01 rad, the smallest std the normaliser divides by, written into `meta/stats.json` by the recorder and by `sim/dataset_view.py` (`floor_std`); `tests/test_record.py` checks it. The smallest real spread is 0.017 rad (start noise), so it changes nothing that varies. Diffusion Policy's normaliser likewise treats near-constant dimensions as constant.
- It matters for every variant: one arm is idle in every primitive, and in T1 each arm is idle for half the episode. Training restarted from scratch on 2026-10-29.

**2. Temporal ensembling (finding; now the default for ACT).** LeRobot's ACT runs each 100-step action chunk open loop by default. After the floor fix the drawer opened by 40k, but the cube slid past 5 mm from jerks where one chunk hands over to the next. ACT's temporal ensembling (a chunk predicted every frame, overlapping predictions averaged with weights exp(−0.01 i), as Zhao et al. 2023 used for their sim results) took the same 60k checkpoint from 4/10 to 10/10. **ACT is evaluated with `--temporal-ensemble 0.01` from here on.** Cost: a forward pass every frame instead of every 100; an S1 rollout takes ~46 s with the GPU to itself, ~75 s while training runs alongside.

**3. Training speed.** At 480 × 640 ACT took 1.6 s per step on this Mac (~45 h per run), nearly all of it the model; at 240 × 320, 0.41 s. Hence the recording size (2data_collection.md, design point 4). The Mac must stay awake (`caffeinate -i -s -w <pid>`) and on mains power: a sleep cost ~1 h, Low Power Mode slows training.

**4. Video decoding.** `torchcodec` can't load with Homebrew FFmpeg 9 (it supports FFmpeg 4–8), so LeRobot falls back to `pyav`. No effect on results.

**5. Task imbalance in the shared scene (research, 2026-10-30; decision for step 3).** With 50 episodes per command, the drawer-closed scene has 50 S1 (stop) and 100 T1 (continue) episodes per opener arm. What similar work does:

| Approach | Used by | Finding |
|---|---|---|
| Record balanced data | LIBERO, Meta-World MT10/50, ACT (equal demos per task) | The reference the others are compared with |
| Upsample the rare behaviour to equal weight (resample or duplicate) | Parekh et al. 2025, "Towards Balanced Behavior Cloning from Imbalanced Datasets" (CALVIN) | An under-represented behaviour loses success; equal-weight upsampling roughly matches a truly balanced dataset, except for a harder behaviour, where the duplicated demos overfit |
| Weights by dataset size, hand-adjusted | Octo, OpenVLA (Open X-Embodiment mixtures) | Built for mixing large datasets |
| Temperature sampling, weight ∝ \|Dᵢ\|^(1/τ) | arXiv 2510.19373 (LIBERO, RoboCasa) | LIBERO 0.76 → 0.85 over size-proportional. With two groups it only interpolates between 1:2 and 1:1 |
| Learned weights (ReMix; meta-gradient) | Hejna et al. CoRL 2024; Parekh et al. | Beat hand-set weights on large mixtures; need reference models; overkill for 6 variants |
| Re-sampling under extreme imbalance (46 vs 5 demos) | arXiv 2602.06512 (miniVLA, LIBERO) | Barely helped (26.5% → 27.1%); the failures were in grounding the object, not the ratio |

- Closest to this case: Parekh et al.'s CALVIN setting, where the same task leads to different behaviours by context, as our command decides stop or continue in one scene. Their result: fix the imbalance, and plain equal-weight upsampling is enough when the data is clean and the rare behaviour isn't harder (here: scripted data, and stopping is the easy behaviour).
- LeRobot 0.6.1 can't weight: `EpisodeAwareSampler` draws every frame uniformly, and `--dataset.episodes` is a mask (an episode listed twice counts once). Upsampling has to be in the data (duplicate episodes in the training view) or a custom sampler.
- These papers weight state-action pairs, not episodes. At the decision state (drawer open, arms home) S1 holds still for 2 s (~100 frames per episode, "stay"), while T1 passes through it in a few frames before the second arm moves. By frames the decision may lean towards stopping, not continuing; count it on the recorded data before choosing a weight.
- **Decision:** record 50 per command (seeds are deterministic, so adding S1 episodes later gives the same episodes as recording them now). Train step 3 first without reweighting, as the baseline; count the stay/move frames at the decision state; then upsample (duplicate episodes in the view) if the stop or continue rate shows a lean.
- **Outcome (step 3):** no lean: no S1 rollout continued and every T1 rollout did (results above), so no reweighting, and the frame count wasn't needed.

**6. SmolVLA needs a rented GPU (measured 2026-11-01).** Fine-tuning `lerobot/smolvla_base` on `data/min` runs on this Mac's GPU (MPS): the camera rename works and the loss falls normally (2.0 → 0.8 in 40 steps). But it takes 4.3 s per step at batch 8 and 17.5 s at batch 32, the same 0.55 s per sample, so the GPU is saturated: LeRobot's recipe (batch 64 × 20k steps = 1.28M samples) would take ~8 days, against ~4 h on one A100 by LeRobot's SmolVLA guide (measured on a rented RTX 4090, Vast.ai, 2026-11-02: 0.87 s per step at batch 64, data loading 0.012 s, peak 20.9 of 24 GB, so the recipe fits as is: ~5 h, ~$1.60). A local AMD RX 6650 XT (8 GB, no official ROCm support) was ruled out: too little memory for the recipe and an estimated few days. Steps for a rented GPU: [4cloud_training.md](4cloud_training.md).

**7. Encoder crash during recording.** A rare SVT-AV1 crash stopped the first step-2 recording; the recorder now retries a failed part ([2data_collection.md](2data_collection.md), "Encoder crash").

**8. Diffusion Policy (step 3b) setup.** LeRobot 0.6.1's DP with its defaults (2 observation frames, horizon 64 and 32 actions executed, about 1.3 s and 0.6 s at 50 fps; ResNet-18 per camera at the full 240 × 320; DDPM with 100 denoising steps), the one-hot input as `observation.environment_state` (added to DP's global conditioning), batch 64 (LeRobot's earlier PushT config and the DP paper's image experiments) and 100k steps (as ACT; ~25 passes over `data/min`). These are our choices, not a published recipe for this setup. DP normalises state and action by their min-max range, not their std, so the normalisation floor now also widens any range below 2 × `STD_FLOOR` (`floor_std`; `data/min` needs none, every joint moves by ≥ 0.86 rad somewhere, but `data/s1a` would). Checked end to end on the Mac (2 training steps, one rollout through `sim/evaluate.py`). Evaluation cost: one S1 rollout took ~146 s on the Mac (100 denoising steps every 32 frames, sharing the GPU with another evaluation), so 300 rollouts would take most of a day.

**9. S1's success rate is set by the slide limit, so step 5 changed (2026-11-02).** Every S1 rollout so far (steps 1 and 3, held-out colours, SmolVLA's first check) opens the drawer and the arms hold still; every failure is the cube sliding past the 5 mm limit, mostly 5.1–5.5 mm. The limit sits just above the expert's largest slide (4.0 mm, median 2.2) and the policies pull about twice as hard, so differences such as 60% vs 84% are mostly where 4–6 mm slides fall against the threshold, not differences in skill. One comparison stays real: S1:A went from 44/50 (step 1) to 30/50 (step 3), a rougher pull. Consequences: no longer ACT run (it would target the threshold, not the task); residual RL, planned as step 5 to fix "precision failures", would mostly learn to beat the same threshold and is parked as optional step 6; step 5 is now stress tests (1tasks_scene.md). S1 apart from the limit (`python -m sim.s1_report RUN...`; task = every other check passes and the arms hold still; @5 mm is the stop rate):

| Policy, condition | Variant | Task | Slide mm (median, p90, max) | @5 mm | @6 mm | @8 mm |
|---|---|---|---|---|---|---|
| Step 1 ACT (S1:A only), train | S1:A | 50/50 | 4.2, 5.2, 7.7 | 44/50 | 48/50 | 50/50 |
| Step 1 ACT, held-out colours | S1:A | 50/50 | 4.3, 5.0, 6.2 | 44/50 | 49/50 | 50/50 |
| Step 3 ACT, train | S1:A | 50/50 | 4.9, 5.7, 6.1 | 30/50 | 48/50 | 50/50 |
| | S1:B | 50/50 | 4.3, 5.1, 6.3 | 42/50 | 49/50 | 50/50 |
| Step 3 ACT, held-out colours | S1:A | 50/50 | 4.8, 5.5, 6.5 | 32/50 | 49/50 | 50/50 |
| | S1:B | 50/50 | 4.4, 5.0, 6.2 | 45/50 | 47/50 | 50/50 |
| Step 4 SmolVLA, train | S1:A | 47/50 | 2.7, 5.2, 8.7 | 39/50 | 45/50 | 46/50 |
| | S1:B | 50/50 | 3.4, 7.2, 8.7 | 36/50 | 42/50 | 48/50 |
| Step 4 SmolVLA, held-out colours | S1:A | 44/50 | 2.7, 6.3, 9.8 | 34/50 | 37/50 | 42/50 |
| | S1:B | 50/50 | 3.7, 7.0, 9.4 | 37/50 | 40/50 | 48/50 |
| Step 4 SmolVLA, held-out wording | S1:A | 10/50 | 4.6, 6.0, 6.0 | 6/50 | 10/50 | 10/50 |
| | S1:B | 37/50 | 2.8, 4.9, 7.4 | 33/50 | 34/50 | 37/50 |

**ACT does S1 every time (400/400) and pulls consistently a little too hard** (median 4.2–4.9 mm, never above 7.7): at 6 mm it succeeds 96–98%, at 8 mm 100%, so step 3's 60% vs 84% on S1:A and S1:B is a 0.6 mm difference in median pull against the threshold. **SmolVLA pulls more gently but less consistently** (median 2.7–3.7 mm, a long tail to 9.8 mm) and misses the task outright in 3–6 of 50 S1:A rollouts (the other arm moves, or the drawer isn't opened). So the 5 mm numbers (SmolVLA ahead on S1:A) and the task numbers (ACT ahead everywhere) point opposite ways; both are reported. The expert, for scale: largest slide 4.0 mm, median 2.2 (finding 9).

**10. Early checks before full evaluations.** `python -m sim.offline_check CKPT DATASET` checks a checkpoint's normalisation stats against the floor and replays recorded training episodes through the policy (its own chunking and history, the recorded observations), in a minute on the CPU. Step 3's ACT: 0.005–0.007 rad mean error per joint. SmolVLA (step 4, 20k): 0.006–0.008 rad, no flags; 5 S1:A rollouts: 4 succeed and stop, the fifth slides the cube 5.5 mm (seed 100000, which ACT fails too); ~60 s per rollout sharing the Mac's GPU (`outputs/min_smolvla/eval_quick`). The DP smoke test on the 4090 caught a missing dependency (`diffusers`; `cloud/train.sh` now installs `lerobot[diffusion]`) before any training time was spent; `scripts/check_dp.sh` runs the same checks on DP's 5k checkpoint.

**11. The idle-arm check misreads a wider start pose (2026-11-03).** `other_arm_stayed_home` fails when any joint of the arm that should stay idle ends more than `IDLE_MOVE` = 0.1 rad from where it started. Training starts are within σ 0.02 rad of the usual pose, so the check is only about the arm leaving. Under `stress_start_pose` (σ 0.06) the idle arm often starts 0.08–0.15 rad off, and both policies, which have only seen it near its usual pose, move it part of the way back: half the S1:A rollouts fail on this check alone, for ACT and SmolVLA alike, tracking the start offset (step 5 results). That is returning towards home, not acting with the wrong arm. Fixed by measuring, as well, how far the idle arm's joints go outside the range between their start and their usual pose (`Monitor.away` in `sim/episode.py`, logged per S1 rollout as `idle_arm_rad`; the check itself is unchanged so earlier runs stay comparable). The rerun with that measure (step 5 results) confirms it: with the arm counted as left only above 0.1 rad on it, S1:A under the wider start pose is 16, 16 and 13/20 for ACT, SmolVLA and DP (task 18, 19, 19/20), against 6, 5 and 8 with the check alone.

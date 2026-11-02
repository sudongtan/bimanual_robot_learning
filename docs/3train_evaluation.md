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


### Minimal plan step 3b: Diffusion Policy with the one-hot input (training, interim checks)

Training on the rented 4090 (`outputs/min_dp`, 100k steps at batch 64, 0.33 s per step, ends ~2026-11-03 00:55). Early checks with `scripts/check_dp.sh` (finding 10): offline replay of 6 training episodes and 3 rollouts each of S1:A and T1:A:left, on the CPU (~2.5 min per S1 rollout, ~4.5 min per T1).

| Checkpoint | Passes over `data/min` | Replay error, mean (worst joint) | S1:A | T1:A:left |
|---|---|---|---|---|
| 5k | ~1.3 | 0.018-0.033 rad (0.04-0.07, the wrist) | 0/3: drawer not opened, the other arm left home | 0/3: cube not at target |
| 20k | ~5 | 0.010-0.016 rad (0.02-0.03) | 0/3: drawer opened, but **the other arm left home** every time (stop 0/3) | 0/3: 2 carried the cube to the left side but 3.5 cm past the spot (limit 2 cm); 1 never started the second arm |

DP is learning (the replay error halved, the drawer now opens, the cube goes to the commanded side). The lead to check at 100k: under the S1 command DP goes on as in T1, which step 3's ACT never did (0/100). A possible reason is how the command enters: ACT gives the one-hot vector its own encoder token, DP appends its 11 numbers to the image and state features in the U-Net's conditioning. Three rollouts are a lead, not a finding.

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

**9. S1's success rate is set by the slide limit, so step 5 changed (2026-11-02).** Every S1 rollout so far (steps 1 and 3, held-out colours, SmolVLA's first check) opens the drawer and the arms hold still; every failure is the cube sliding past the 5 mm limit, mostly 5.1–5.5 mm. The limit sits just above the expert's largest slide (4.0 mm, median 2.2) and the policies pull about twice as hard, so differences such as 60% vs 84% are mostly where 4–6 mm slides fall against the threshold, not differences in skill. One comparison stays real: S1:A went from 44/50 (step 1) to 30/50 (step 3), a rougher pull. Consequences: no longer ACT run (it would target the threshold, not the task); residual RL, planned as step 5 to fix "precision failures", would mostly learn to beat the same threshold and is parked as optional step 6; step 5 is now stress tests (1tasks_scene.md). Still to do: report S1 as the task rate (opened and held still) with the slide in mm and success at 5/6/8 mm.

**10. Early checks before full evaluations.** `python -m sim.offline_check CKPT DATASET` checks a checkpoint's normalisation stats against the floor and replays recorded training episodes through the policy (its own chunking and history, the recorded observations), in a minute on the CPU. Step 3's ACT: 0.005–0.007 rad mean error per joint. SmolVLA (step 4, 20k): 0.006–0.008 rad, no flags; 5 S1:A rollouts: 4 succeed and stop, the fifth slides the cube 5.5 mm (seed 100000, which ACT fails too); ~60 s per rollout sharing the Mac's GPU (`outputs/min_smolvla/eval_quick`). The DP smoke test on the 4090 caught a missing dependency (`diffusers`; `cloud/train.sh` now installs `lerobot[diffusion]`) before any training time was spent; `scripts/check_dp.sh` runs the same checks on DP's 5k checkpoint.

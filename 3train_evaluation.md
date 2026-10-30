## Training and evaluation

Results per plan step ([1tasks_scene.md](1tasks_scene.md), "Minimal plan"), and the bugs and findings that changed how policies are trained or evaluated. Recording design and data checks are in [2data_collection.md](2data_collection.md).

**How a number is made.**
- Training: `lerobot-train` (LeRobot 0.6.1), ACT at LeRobot's setting (100k steps, batch 8), on this Mac's GPU (`mps`) at 240 × 320 (2data_collection.md, design point 4), ~2.4 steps/s, ~12 h per run. Script: `outputs/train_step1.sh`; for a rented GPU, `cloud/train.sh`.
- Evaluation: `python -m sim.evaluate CKPT --variants S1:A --episodes 50 --temporal-ensemble 0.01`. Rollouts start from `EVAL_SEEDS` (the same for every variant and model, never trained on), run a fixed horizon (the task's longest demo plus 2 s, as CALVIN) and are scored by the expert's own success checks (`sim/episode.py`), plus the stop rate (S1, S3) and continue rate (T1, T2). Each rollout logs its checks and the cube's slide in the drawer (`cube_slide_mm`) to `episodes.jsonl`.

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

**Still to run:** the 100k `action` checkpoint on held-out colours (`--condition eval_colours`). With 0.2–2 mm of slide to spare, some rollouts may tip over the limit there for reasons unrelated to colour; read failures by check, not just the rate.

## Findings and bugs

**1. Normalisation floor (bug, fixed).** LeRobot normalises each joint by its standard deviation over the dataset. In S1 arm A the idle arm's gripper never moves (std 7 × 10⁻⁹), so physics jitter of ~3 × 10⁻⁷ rad in a rollout became normalised inputs of 15–50 (all others within ±3), and the network could tell training episodes apart by that jitter. ACT fitted the recorded actions to 0.003–0.009 rad offline but opened the drawer in 0/10 eval starts and 2/10 training starts at 40k steps (`outputs/s1a_act_action_nofloor`). Ruled out first: the rollout loop (same first observation and physics state as the recording), image compression (rendered and video-decoded images give equally accurate predictions), and temporal ensembling on its own (1/8).
- Fix: `STD_FLOOR` = 0.01 rad, the smallest std the normaliser divides by, written into `meta/stats.json` by the recorder and by `sim/dataset_view.py` (`floor_std`); `tests/test_record.py` checks it. The smallest real spread is 0.017 rad (start noise), so it changes nothing that varies. Diffusion Policy's normaliser likewise treats near-constant dimensions as constant.
- It matters for every variant: one arm is idle in every primitive, and in T1 each arm is idle for half the episode. Training restarted from scratch on 2026-10-29.

**2. Temporal ensembling (finding; now the default for ACT).** LeRobot's ACT runs each 100-step action chunk open loop by default. After the floor fix the drawer opened by 40k, but the cube slid past 5 mm from jerks where one chunk hands over to the next. ACT's temporal ensembling (a chunk predicted every frame, overlapping predictions averaged with weights exp(−0.01 i), as Zhao et al. 2023 used for their sim results) took the same 60k checkpoint from 4/10 to 10/10. **ACT is evaluated with `--temporal-ensemble 0.01` from here on.** Cost: a forward pass every frame instead of every 100; an S1 rollout takes ~46 s with the GPU to itself, ~75 s while training runs alongside.

**3. Training speed.** At 480 × 640 ACT took 1.6 s per step on this Mac (~45 h per run), nearly all of it the model; at 240 × 320, 0.41 s. Hence the recording size (2data_collection.md, design point 4). The Mac must stay awake (`caffeinate -i -s -w <pid>`) and on mains power: a sleep cost ~1 h, Low Power Mode slows training.

**4. Video decoding.** `torchcodec` can't load with Homebrew FFmpeg 9 (it supports up to 7.x), so LeRobot falls back to `pyav`. No effect on results.

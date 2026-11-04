# Status and handoff

_Last updated: 2026-11-04. Update this file whenever a job starts or ends._

**The project is complete (2026-11-04):** the minimal plan (1tasks_scene.md) is done for ACT, Diffusion Policy and SmolVLA, and ends there. Every experiment not run is listed under "Planned work" in ../README.md. This file records the final state of the runs.

## Done (minimal plan, 1tasks_scene.md)

| Step | Result | Details |
|---|---|---|
| 1 | ACT on S1:A: 88% (44/50) train colours, 88% (44/50) held-out colours; held still 50/50 | 3train_evaluation.md |
| 2 | `data/min` recorded and checked (300 episodes) | 2data_collection.md |
| 3 | ACT with the one-hot input, 6 variants: S1 72/100, T1 195/200; the command decides stop/continue and target every time. Held-out colours: S1 77/100, T1 194/200, no colour gap | 3train_evaluation.md |
| 4 (trained) | SmolVLA, 20k steps on the 4090 (final loss 0.007). First check: 4/5 S1:A rollouts succeed (the failure is the slide), offline replay error 0.006–0.008 rad (ACT: 0.005–0.007) | `outputs/min_smolvla/eval_quick` |
| 4 (evaluated) | SmolVLA, 6 variants × 50 under train / held-out colours / held-out wording: S1 76 / 71 / 49 of 100, T1 111 / 122 / 0 of 200 (ACT: S1 72, T1 195 on train). T1 misses are placement (it continues 198/200); unseen wording breaks it (3train_evaluation.md, step 4) | `outputs/min_smolvla/eval` |
| 3b (trained) | DP, 100k steps on the 4090 (resumed from 50k after the disk filled). 100k check: offline replay 0.002–0.003 rad, no flags; S1:A 4/5 (the other arm still 5/5), T1:A:left 5/5. 10 DDIM steps replay as well as the 100 DDPM steps | `outputs/min_dp/eval_100000` |
| 3b (evaluated) | DP, 10 DDIM steps (checked against 100 DDPM: S1 9/10 both, T1 8/10 against 10/10), 6 variants × 50: train S1 82/100, T1 181/200; held-out colours S1 81/100, T1 165/200. Follows the command every time; T1 misses are placement | 3train_evaluation.md, step 3b |
| 5 (ACT, SmolVLA, DP) | Stress tests, S1:A and T1:A:left × 20: S1 on the task unaffected for all three (start pose with the corrected idle-arm measure, finding 11); T1: ACT holds (17–19/20), DP drops 17 → 10–12, SmolVLA not comparable (12/20 unchanged) | 3train_evaluation.md, step 5 |

## Running now

Nothing is running on the Mac or the rented GPU (as of 2026-11-04 07:30 CET).

**The rented machine:** reached as `ssh gpu` (alias in this Mac's `~/.ssh/config`: 195.224.35.250, port 20412, user root). Vast's login opens its own tmux, so watch a session with `ssh -t gpu "tmux attach -t train"` (leave with `Ctrl-b d`, never `Ctrl-c`). Files are in `/root/bimanual`. It bills about $0.33/h until it is **deleted** in the Vast dashboard (the user deletes it). Idle since DP ended (2026-11-03 02:52 CET); everything needed is copied to this Mac, so the instance can be deleted if it still exists (the user deletes it).

## Planned (not run)

Listed with the reason for each under "Planned work" in ../README.md: ACT and DP reading the text commands (design in 3train_evaluation.md); a SmolVLA re-planning test for its T1 placement (`n_action_steps` 10, ~40 min); why DP's placement drops under the stress changes; the stress factors that need code (cube start past the training range, camera pose, mid-episode disturbance); more training seeds; the full plan's other tasks and π0 (1tasks_scene.md); residual RL (`sim/residual.py`). If any is run, record it in 3train_evaluation.md and the status lines in README.md, 1tasks_scene.md and this file.

## Open questions for the user

- Decided 2026-11-02: no longer ACT run. S1's failures are all slides just past the 5 mm limit (every rollout opens the drawer), so more training targets the threshold, not the task. Extra training seeds (error bars) remain an option.
- S1 reported apart from the 5 mm limit (finding 9, `python -m sim.s1_report`): done for ACT, SmolVLA and DP.
- `pyproject.toml` lists a `policies/lerobot_policy_bc_mlp` plugin that doesn't exist (why `uv run` fails). Left as is.

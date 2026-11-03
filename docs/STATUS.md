# Status and handoff

_Last updated: 2026-11-03 15:45 CET. Update this file whenever a job starts or ends._

Where the project stands, what is running right now, and what comes next. The design is in the numbered docs (../README.md, "Documents"); this file is only the live state.

## Done (minimal plan, 1tasks_scene.md)

| Step | Result | Details |
|---|---|---|
| 1 | ACT on S1:A: 88% (44/50) train colours, 88% (44/50) held-out colours; held still 50/50 | 3train_evaluation.md |
| 2 | `data/min` recorded and checked (300 episodes) | 2data_collection.md |
| 3 | ACT with the one-hot input, 6 variants: S1 72/100, T1 195/200; the command decides stop/continue and target every time. Held-out colours: S1 77/100, T1 194/200, no colour gap | 3train_evaluation.md |
| 4 (trained) | SmolVLA, 20k steps on the 4090 (final loss 0.007). First check: 4/5 S1:A rollouts succeed (the failure is the slide), offline replay error 0.006–0.008 rad (ACT: 0.005–0.007) | `outputs/min_smolvla/eval_quick` |
| 4 (evaluated) | SmolVLA, 6 variants × 50 under train / held-out colours / held-out wording: S1 76 / 71 / 49 of 100, T1 111 / 122 / 0 of 200 (ACT: S1 72, T1 195 on train). T1 misses are placement (it continues 198/200); unseen wording breaks it (3train_evaluation.md, step 4) | `outputs/min_smolvla/eval` |
| 3b (trained) | DP, 100k steps on the 4090 (resumed from 50k after the disk filled). 100k check: offline replay 0.002–0.003 rad, no flags; S1:A 4/5 (the other arm still 5/5), T1:A:left 5/5. 10 DDIM steps replay as well as the 100 DDPM steps | `outputs/min_dp/eval_100000` |
| 5 (ACT, SmolVLA) | Stress tests, S1:A and T1:A:left × 20: distractors no effect; turned cube costs SmolVLA's T1 (9 against 12 on the same starts), not ACT; the start-pose S1 drop is the idle-arm check (finding 11); SmolVLA: more phrasings S1 10/20, T1 0/20, empty command 0/20 (arm B moves) | 3train_evaluation.md, step 5 |

## Running now

| Job | Where | Started | Expected end | Check with |
|---|---|---|---|---|
| DP evaluation (`scripts/eval_dp.sh`, 10 DDIM steps): train and eval_colours 6 × 50, stress S1:A and T1:A:left × 20, then 10 + 10 with the 100 DDPM steps as a check | Mac GPU, nohup | 2026-11-03 14:32 CET | ~2026-11-04 04:00–04:30 CET (~39 s per S1 rollout with DDIM) | `cat outputs/eval_dp.out` |
| Start-pose rerun (finding 11): ACT and SmolVLA under `stress_start_pose`, S1:A and T1:A:left × 20, now logging the idle arm's distance outside its start–home range (`idle_arm_rad.away`) | Mac GPU, nohup (`scripts/rerun_start_pose.sh`), waits for DP | queued 2026-11-03 15:40 CET | ~1.5 h after DP, ~2026-11-04 06:00 CET | `cat outputs/rerun_start_pose.out` |

**The rented machine:** reached as `ssh gpu` (alias in this Mac's `~/.ssh/config`: 195.224.35.250, port 20412, user root). Vast's login opens its own tmux, so watch a session with `ssh -t gpu "tmux attach -t train"` (leave with `Ctrl-b d`, never `Ctrl-c`). Files are in `/root/bimanual`. It bills about $0.33/h until it is **deleted** in the Vast dashboard (the user deletes it). Idle since DP ended (2026-11-03 02:52 CET); everything needed is copied to this Mac.

## Next

1. SmolVLA evaluated (3train_evaluation.md, step 4). Open: whether its loose T1 placement is the policy or the 50-action open-loop execution (a re-planning test, e.g. `n_action_steps` 10, ~40 min; not run).
2. DP trained (100k, `outputs/min_dp/checkpoints/100000`; ended 2026-11-03 02:52 CET, copy verified; the user can delete the Vast instance). 100k check (3train_evaluation.md, step 3b): no flags, replay 0.002-0.003 rad, S1:A 4/5 with the other arm still in 5/5, T1:A:left 5/5. Decided with the user: the full evaluation with 10 DDIM steps (`--ddim 10`; offline replay error unchanged), checked against the 100 DDPM steps on 20 rollouts (`scripts/eval_dp.sh`, queued).
3. Step 5 (stress tests, 1tasks_scene.md): DP's stress tests are in `scripts/eval_dp.sh`; then the factors that need code (cube start past the training range, camera pose, mid-episode disturbance). Residual RL is parked as optional step 6, only for a gap step 5 finds.
4. Record each result in 3train_evaluation.md and the status lines in README.md, 1tasks_scene.md and this file.

## Open questions for the user

- Decided 2026-11-02: no longer ACT run. S1's failures are all slides just past the 5 mm limit (every rollout opens the drawer), so more training targets the threshold, not the task. Extra training seeds (error bars) remain an option.
- S1 reported apart from the 5 mm limit (finding 9, `python -m sim.s1_report`): done 2026-11-03; add DP when its evaluation ends.
- `pyproject.toml` lists a `policies/lerobot_policy_bc_mlp` plugin that doesn't exist (why `uv run` fails). Left as is.

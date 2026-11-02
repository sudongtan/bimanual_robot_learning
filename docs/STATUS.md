# Status and handoff

_Last updated: 2026-11-02 18:45 CET. Update this file whenever a job starts or ends._

Where the project stands, what is running right now, and what comes next. The design is in the numbered docs (../README.md, "Documents"); this file is only the live state.

## Done (minimal plan, 1tasks_scene.md)

| Step | Result | Details |
|---|---|---|
| 1 | ACT on S1:A: 88% (44/50) train colours, 88% (44/50) held-out colours; held still 50/50 | 3train_evaluation.md |
| 2 | `data/min` recorded and checked (300 episodes) | 2data_collection.md |
| 3 | ACT with the one-hot input, 6 variants: S1 72/100, T1 195/200; the command decides stop/continue and target every time. Held-out colours: S1 77/100, T1 194/200, no colour gap | 3train_evaluation.md |
| 4 (trained) | SmolVLA, 20k steps on the 4090 (final loss 0.007). First check: 4/5 S1:A rollouts succeed (the failure is the slide), offline replay error 0.006–0.008 rad (ACT: 0.005–0.007) | `outputs/min_smolvla/eval_quick` |

## Running now

| Job | Where | Started | Expected end | Check with |
|---|---|---|---|---|
| DP training (step 3b), 100k steps, batch 64 (smoke test: 0.33 s/step, 21.3 GB peak) | Rented RTX 4090 (Vast.ai instance 58799630), tmux session `dp` | 2026-11-02 15:42 CET | ~2026-11-03 00:55 CET | `ssh gpu "tail -c 200 bimanual/min_dp.log"` |
| SmolVLA evaluation (step 4): train, eval_colours, eval_wording, 6 variants × 50 each | This Mac, detached (`scripts/eval_smolvla.sh`) | 2026-11-02 18:19 CET | ~2026-11-03 08:00 CET (~55 s per S1 rollout, ~95 s per T1) | `cat outputs/eval_smolvla.out; tail -3 outputs/min_smolvla_*.log` |
| Step 5 stress tests: ACT (step 3) and SmolVLA under `stress_distractors`, `stress_start_pose`, `stress_cube_yaw`, S1:A and T1:A:left × 20 each | This Mac, detached (`scripts/stress.sh`), waits for the SmolVLA evaluation | queued | ~4–5 h after it starts (240 rollouts at ~65 s), ~2026-11-03 13:00 CET | `cat outputs/stress.out` |

**The rented machine:** reached as `ssh gpu` (alias in this Mac's `~/.ssh/config`: 195.224.35.250, port 20412, user root). Vast's login opens its own tmux, so watch a session with `ssh -t gpu "tmux attach -t train"` (leave with `Ctrl-b d`, never `Ctrl-c`). Files are in `/root/bimanual`. It bills about $0.33/h until it is **deleted** in the Vast dashboard (the user deletes it).

## Next

1. SmolVLA's copy-back and evaluation are automated (`scripts/eval_smolvla.sh`, above; results in `outputs/min_smolvla/eval/<condition>/`). Record them in 3train_evaluation.md when done.
2. When DP ends: copy it back the same way (`outputs/min_dp`), then ask the user to delete the instance. Interim checks at 5k and 20k (3train_evaluation.md, step 3b): learning, but at 20k the S1 command doesn't stop it (the other arm leaves home in 3/3); check that first at 100k (`zsh scripts/check_dp.sh 100000 5`, CPU). DP evaluation is slow on the Mac (~146 s per S1 rollout, ~a day for 300); options to decide with the user: fewer denoising steps at inference (DDIM; a method change to report) or evaluating on the rented machine (untested headless MuJoCo rendering).
3. Step 5 (stress tests, 1tasks_scene.md): add DP to `scripts/stress.sh` once it is trained; then the factors that need code (cube start past the training range, camera pose, mid-episode disturbance). Residual RL is parked as optional step 6, only for a gap step 5 finds.
4. Record each result in 3train_evaluation.md and the status lines in README.md, 1tasks_scene.md and this file.

## Open questions for the user

- Decided 2026-11-02: no longer ACT run. S1's failures are all slides just past the 5 mm limit (every rollout opens the drawer), so more training targets the threshold, not the task. Extra training seeds (error bars) remain an option.
- Report S1 more honestly: the task rate (drawer opened, arms held still) with the slide in mm and success at 5/6/8 mm, from the logged rollouts. Not done yet.
- `pyproject.toml` lists a `policies/lerobot_policy_bc_mlp` plugin that doesn't exist (why `uv run` fails). Left as is.

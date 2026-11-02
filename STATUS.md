# Status and handoff

_Last updated: 2026-11-02 11:55 CET. Update this file whenever a job starts or ends._

Where the project stands, what is running right now, and what comes next. The design is in the numbered docs (README.md, "Documents"); this file is only the live state.

## Done (minimal plan, 1tasks_scene.md)

| Step | Result | Details |
|---|---|---|
| 1 | ACT on S1:A: 88% (44/50) train colours, 88% (44/50) held-out colours; held still 50/50 | 3train_evaluation.md |
| 2 | `data/min` recorded and checked (300 episodes) | 2data_collection.md |
| 3 | ACT with the one-hot input, 6 variants: S1 72/100, T1 195/200; the command decides stop/continue and target every time | 3train_evaluation.md |

## Running now

| Job | Where | Started | Expected end | Check with |
|---|---|---|---|---|
| SmolVLA training (step 4), 20k steps, batch 64 | Rented RTX 4090 (Vast.ai instance 58799630), tmux session `train` | 2026-11-02 09:41 UTC | ~14:40 UTC | `ssh gpu "grep -o 'step:[0-9KM]* .*updt_s:[0-9.]*' bimanual/min_smolvla.log \| tail -1"` |
| DP chain (step 3b): waits for `train` to end, then DP smoke test, then DP training (100k steps, batch 64) | Same machine, tmux session `dp`, script `~/bimanual/dp_after_smolvla.sh` | queued | smoke ~14:50 UTC, training a few hours after (time printed by the smoke test) | `ssh gpu "cat bimanual/dp_chain.log bimanual/smoke_dp.out"` |
| Step 3 ACT on held-out colours, 6 variants × 50 | This Mac, detached (`scripts/eval_colours_step3.sh`) | 2026-11-02 11:51 CET | ~17:45 CET | `tail -3 outputs/min_eval_colours.log` |

**The rented machine:** reached as `ssh gpu` (alias in this Mac's `~/.ssh/config`: 195.224.35.250, port 20412, user root). Vast's login opens its own tmux, so watch a session with `ssh -t gpu "tmux attach -t train"` (leave with `Ctrl-b d`, never `Ctrl-c`). Files are in `/root/bimanual`. It bills about $0.33/h until it is **deleted** in the Vast dashboard (the user deletes it).

## Next

1. When SmolVLA ends: copy `checkpoints/last/pretrained_model` back (cloud/README.md, step 10), then evaluate it on the Mac under train, eval_colours and eval_wording (step 12) once the colour run above has finished (one GPU job at a time on the Mac).
2. When DP ends: copy it back the same way (`outputs/min_dp`), then ask the user to delete the instance. DP evaluation is slow on the Mac (~146 s per S1 rollout, ~a day for 300); options to decide with the user: fewer denoising steps at inference (DDIM; a method change to report) or evaluating on the rented machine (untested headless MuJoCo rendering).
3. Record each result in 3train_evaluation.md and the status lines in README.md, 1tasks_scene.md and this file.

## Open questions for the user

- Train ACT longer (step 3 saw each frame ~3 times, step 1 ~30) to test whether S1's lower success in step 3 is undertraining. Suggested on the rented 4090 after DP; not decided.
- `pyproject.toml` lists a `policies/lerobot_policy_bc_mlp` plugin that doesn't exist (why `uv run` fails). Left as is.

# Training SmolVLA on a rented GPU: step by step

Minimal-plan step 4: fine-tune SmolVLA on `data/min` on a rented GPU, then evaluate it on the Mac.

- **Why rent:** LeRobot's recipe (batch 64, 20k steps) would take ~8 days on the Mac. It takes ~4 h on one A100 by LeRobot's SmolVLA guide, or an estimated 5–7 h on an RTX 4090 ([3train_evaluation.md](3train_evaluation.md), finding 6).
- **What runs where:** training needs only LeRobot and the dataset. Evaluation needs MuJoCo and `sim/`, so it stays on the Mac.
- **Expect:** ~30 min of your time, plus 4–7 h of training. The cost is a few dollars at typical 4090/A100 rates; check the provider's current prices.

Commands marked **Mac** run in a terminal at the repository root (`~/Documents/bimanual`). Commands marked **GPU** run on the rented machine, over SSH.

---

## Step 1. Make sure you have an SSH key (once)

**Mac:**

```
ls ~/.ssh/id_ed25519.pub
```

If it says "No such file", create one (press Enter at every prompt):

```
ssh-keygen -t ed25519
```

Print the public key and copy it:

```
cat ~/.ssh/id_ed25519.pub
```

## Step 2. Rent a machine

1. Make an account with a GPU provider that gives SSH access (RunPod, Vast.ai, Lambda, or a university cluster).
2. Paste your public key from step 1 into the account's SSH key settings.
3. Start a machine with:

| | Choose | Why |
|---|---|---|
| GPU | 1 × RTX 4090 (24 GB) or 1 × A100 (40/80 GB) | 24 GB may need a smaller batch (step 8) |
| Disk | ≥ 80 GB | Dataset 2.5 GB, Python environment ~10 GB, checkpoints: every 5k steps LeRobot saves the model and the optimizer state (DP: 1.1 + 2.2 GB), so 20 DP checkpoints need ~66 GB. On 2026-11-02 a 50 GB disk filled at DP's 55k save; `cloud/dp_resume.sh` resumes and keeps optimizer state for the latest checkpoint only |
| CPU | ≥ 8 cores | Decoding three video streams per sample |
| Image / template | Any recent PyTorch or CUDA Ubuntu image | `train.sh` installs its own Python 3.12, which LeRobot 0.6.1 needs |

4. When it's running, find its SSH command in the dashboard. It looks like `ssh -p 40022 root@203.0.113.7`.

## Step 3. Give the machine a short name

**Mac:** open `~/.ssh/config` (e.g. `open -e ~/.ssh/config`; create it if it doesn't exist) and add these lines, using the address and port from step 2:

```
Host gpu
    HostName 203.0.113.7
    Port 40022
    User root
```

Test it:

```
ssh gpu nvidia-smi
```

- **Expect:** a table showing the GPU (answer `yes` the first time it asks about the host).
- **If it fails:** see Troubleshooting below.

## Step 4. Prepare the GPU machine

**Mac:**

```
ssh gpu "apt-get update && apt-get install -y rsync tmux curl && mkdir -p bimanual/data"
```

This installs the three tools the next steps use, which minimal images may lack. If it says `apt-get: command not found` or `permission denied`, see Troubleshooting.

## Step 5. Copy the dataset and the training script

**Mac:**

```
rsync -aL --exclude sim --info=progress2 data/min gpu:bimanual/data/
rsync -a cloud/train.sh cloud/smoke_test.sh gpu:bimanual/
```

- **Expect:** a progress line up to ~2.5 GB, a few minutes.
- `--exclude sim` leaves out the recorder's states, which training doesn't need.

## Step 6. Start training

**Mac:** start the smoke test in a background `tmux` session on the GPU machine:

```
ssh gpu "cd bimanual && tmux new -d -s smoke 'bash smoke_test.sh > smoke.out 2>&1'"
```

Starting `tmux` from the Mac (`-d`, detached) works whether or not the login opens a `tmux` of its own. Vast's images do: a plain `tmux new` or `tmux attach` typed inside such a login fails with "sessions should be nested with care". To watch a session from the Mac, use `ssh -t gpu "tmux attach -t smoke"` (leave with `Ctrl-b`, then `d`; `Ctrl-c` would stop it). From inside a Vast login, use `tmux switch-client -t smoke`. Check the result with `ssh gpu "cat bimanual/smoke.out"`.

The smoke test (~5–10 min) runs 30 training steps at batch 64, and at batch 32 if 64 runs out of memory. That checks the install, the model download, video decoding, GPU memory and checkpoint saving. It ends with either:
- `OK at batch ...`, the time per step, peak GPU memory and the full run's estimated hours, plus a `NEXT:` line with the exact command for the real run; or
- `FAIL ...` and the last lines of the log (see Troubleshooting).

Then run the `NEXT:` command in its own session. For batch 64, from the Mac:

```
ssh gpu "cd bimanual && tmux new -d -s train 'bash train.sh smolvla data/min outputs/min_smolvla 2>&1 | tee min_smolvla.log'"
```

- **Setup:** the first command run (the smoke test) installs `uv`, Python 3.12 and LeRobot and downloads `lerobot/smolvla_base` (public, no Hugging Face login needed); later runs reuse them.
- **Then training starts:** lines like `step:500 ... loss:... updt_s:...` appear.
- **It keeps running** in the detached session, even if you close the terminal or the Mac sleeps.

## Step 7. Check it after ~15 minutes

**Mac:**

```
ssh gpu "grep -o 'step:[0-9KM]* .*updt_s:[0-9.]*' bimanual/min_smolvla.log | tail -3"
ssh gpu nvidia-smi
```

Read the output:
- **Time:** total ≈ `updt_s` × 20,000 seconds. At `updt_s:0.72`, that's ~4 h.
- **Loss:** it should be falling (on the Mac it went 2.0 → 0.8 in 40 steps).
- **Data loading:** if GPU use in `nvidia-smi` stays low while the steps are slow, the CPU can't decode video fast enough. Rent a machine with more cores.
- **Memory:** if the log shows `CUDA out of memory`, go to step 8.

To watch it live: `ssh -t gpu "tmux attach -t train"` (detach again with `Ctrl-b`, then `d`).

## Step 8. Only if it ran out of memory

**GPU** (`ssh -t gpu "tmux attach -t train"`): press `Ctrl-c`, then run half the batch for twice the steps (the same number of samples):

```
rm -rf outputs/min_smolvla
bash train.sh smolvla data/min outputs/min_smolvla --batch_size=32 --steps=40000 2>&1 | tee min_smolvla.log
```

The old output folder has to go first, because LeRobot refuses to write into an existing one.

## Step 9. Wait for it to finish

**Mac:** check every few hours:

```
ssh gpu "tail -c 300 bimanual/min_smolvla.log"
```

It's done when the log ends with `End of training`.

## Step 10. Copy the trained model back

**Mac:**

```
mkdir -p outputs/min_smolvla/checkpoints/last
rsync -aL gpu:bimanual/outputs/min_smolvla/checkpoints/last/pretrained_model outputs/min_smolvla/checkpoints/last/
rsync -a gpu:bimanual/min_smolvla.log outputs/
ls outputs/min_smolvla/checkpoints/last/pretrained_model
```

- **Expect:** `config.json`, `model.safetensors`, `train_config.json` and the `policy_preprocessor*` / `policy_postprocessor*` files. These are the weights, plus the normalisation and camera renaming that evaluation needs.

## Step 11. Delete the machine

In the provider's dashboard, **delete (terminate)** the instance once step 10 has worked. Most providers bill until it's deleted, and some bill for the disk while it's only stopped.

## Step 12. Evaluate on the Mac

**Mac:** use the project's `.venv` Python. First, one rollout to check it loads and to time it:

```
python -m sim.evaluate outputs/min_smolvla/checkpoints/last/pretrained_model --variants S1:A --episodes 1
```

If it fails on the Mac's GPU, add `--device cpu`. That's slower: about 4 min per rollout was measured earlier.

Then the full evaluation: 6 variants × 50 rollouts × 3 conditions. There's no `--temporal-ensemble` here; that's for ACT only.

```
V=S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right
C=outputs/min_smolvla/checkpoints/last/pretrained_model
for cond in train eval_colours eval_wording; do
  caffeinate -i -s python -m sim.evaluate $C --variants $V --episodes 50 --videos 2 --condition $cond --out outputs/min_smolvla/eval
done
```

Multiply the one-rollout time by 900 to see how long this will run. `caffeinate` keeps the Mac awake; keep it on mains power.

## Step 13. Read the results

- **Where:** `outputs/min_smolvla/eval/<condition>/summary.json` gives success, stop rate (S1) and continue rate (T1) per variant; `episodes.jsonl` gives every rollout.
- **Compare the train condition with step 3's ACT** (`outputs/min_act/eval_100000_te/train/summary.json`). The gap is the cost of reading the command in language instead of a one-hot input.
- **Record them:** add the results to [3train_evaluation.md](3train_evaluation.md) under a step 4 section.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ssh: connect ... Connection refused` or a timeout | The machine is still starting, or the port changed: check the SSH command in the dashboard and update `~/.ssh/config` |
| `Permission denied (publickey)` | The provider doesn't have your key: add `~/.ssh/id_ed25519.pub` to the account (step 1), then restart the machine if the provider asks |
| `apt-get: command not found` or `permission denied` on `apt-get` | Not root on Ubuntu: prefix with `sudo`, or ask the cluster admin. Without `tmux`, start training with `nohup bash train.sh smolvla data/min outputs/min_smolvla > min_smolvla.log 2>&1 &` and follow it with `tail -f min_smolvla.log` |
| `rsync: command not found` | Run step 4 (rsync must be on both machines; the Mac has it) |
| `uv` or Python install fails | Check that `curl` is installed (step 4) and the machine has internet access |
| `Output directory ... already exists and resume is False` | `rm -rf outputs/min_smolvla` on the GPU machine, then start again |
| `CUDA out of memory` | Step 8 |
| Download from Hugging Face fails | Retry; if it's rate-limited, log in on the GPU machine with a free token: `source .venv-train/bin/activate && hf auth login` |
| Training stopped when SSH dropped | It wasn't inside `tmux`: start again inside `tmux` (step 6) |

## Also possible: Diffusion Policy (step 3b) or ACT on the same machine

Diffusion Policy uses the one-hot view, like ACT. The view links `data/min`'s videos, so either copy it with `-L`, or, if `data/min` is already on the machine, copy it without videos and link them there:

```
rsync -a --exclude videos --exclude sim data/min_onehot gpu:bimanual/data/                            # Mac
ssh gpu "ln -s ~/bimanual/data/min/videos ~/bimanual/data/min_onehot/videos"                          # Mac
ssh gpu "cd bimanual && tmux new -d -s dp 'bash smoke_test.sh dp data/min_onehot > smoke_dp.out 2>&1 && bash smoke_next_dp.sh'"
```

Evaluate it without `--temporal-ensemble` (ACT only).

### ACT

For example, to test whether step 3's ACT was undertrained by training it 3× longer. The one-hot view links `data/min`'s videos, so copy with `-L`.

**Mac:**

```
rsync -aL --exclude sim data/min_onehot gpu:bimanual/data/
```

**GPU** (inside `tmux`):

```
bash train.sh act data/min_onehot outputs/min_act_long --steps=300000 2>&1 | tee min_act_long.log
```

Copy it back as in step 10, and evaluate with `--temporal-ensemble 0.01`, as every ACT result so far.

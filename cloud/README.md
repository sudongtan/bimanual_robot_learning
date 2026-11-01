# Training on a rented GPU

SmolVLA's fine-tuning recipe (LeRobot: batch 64, 20k steps) would take ~8 days on the Mac, against ~4 h on one A100 by LeRobot's SmolVLA guide, or an estimated 5–7 h on an RTX 4090 ([3train_evaluation.md](../3train_evaluation.md), finding 6). Training needs only LeRobot and the dataset, so it runs on any CUDA machine. Evaluation needs MuJoCo and the `sim/` code, so it stays on the Mac.

These steps are for minimal-plan step 4 (SmolVLA on `data/min`). ACT runs the same way (step 9).

## 1. Rent a machine

Any provider that gives SSH access to a Linux GPU machine (RunPod, Vast.ai, Lambda, a university cluster).

| | Minimum | Why |
|---|---|---|
| GPU | RTX 4090 (24 GB) or A100 (40/80 GB) | 24 GB may need batch 32 (step 5) |
| Disk | 50 GB | Dataset 2.5 GB, Python environment ~10 GB, a checkpoint every 5k steps |
| CPU | 8 cores or more | Decoding three video streams per sample; `train.sh` uses 8 data workers |
| Image | Any recent CUDA image (PyTorch or plain Ubuntu) | PyTorch from PyPI needs a recent NVIDIA driver. `train.sh` installs its own Python 3.12 (LeRobot 0.6.1 needs ≥ 3.12) |

Note the SSH command the provider gives you, e.g. `ssh -p 40022 root@203.0.113.7`. Below, `HOST` stands for `-p 40022 root@203.0.113.7`. Simplest is an entry in `~/.ssh/config` on the Mac:

```
Host gpu
    HostName 203.0.113.7
    Port 40022
    User root
```

Then `HOST` is just `gpu`.

## 2. Copy the dataset and the script (on the Mac)

From the repository root:

```
ssh gpu "mkdir -p bimanual/data"
rsync -aL --exclude sim --info=progress2 data/min gpu:bimanual/data/
rsync -a cloud/train.sh gpu:bimanual/
```

`--exclude sim` leaves out the recorder's states (not needed for training). It takes a few minutes for 2.5 GB.

## 3. Start training (on the GPU machine)

```
ssh gpu
cd bimanual
tmux new -s train
bash train.sh smolvla data/min outputs/min_smolvla 2>&1 | tee min_smolvla.log
```

`tmux` keeps the run going if the SSH connection drops: detach with `Ctrl-b d`, and come back later with `ssh gpu` then `tmux attach -t train`. The first run installs `uv`, Python 3.12 and LeRobot (a few minutes), then downloads `lerobot/smolvla_base` from Hugging Face. No login is needed; the model is public.

## 4. Check the first few hundred steps

In a second terminal (`ssh gpu`):

```
grep -o "step:[0-9KM]* .*updt_s:[0-9.]*" bimanual/min_smolvla.log | tail -3
nvidia-smi
```

- `updt_s` is the time per step. Total time ≈ `updt_s` × 20,000. At ~0.7 s per step, that's ~4 h.
- `loss` should fall within the first few hundred steps (on the Mac: 2.0 → 0.8 in 40 steps).
- If the step time looks long but GPU use in `nvidia-smi` is low, data loading is the bottleneck: rent a machine with more CPU cores.

## 5. If it runs out of memory

`CUDA out of memory` in the log, usually on a 24 GB card: stop with `Ctrl-c` and restart with half the batch and twice the steps (the same number of samples). The new run needs a fresh output directory:

```
rm -rf outputs/min_smolvla
bash train.sh smolvla data/min outputs/min_smolvla --batch_size=32 --steps=40000 2>&1 | tee min_smolvla.log
```

Extra arguments override `train.sh`'s defaults (the last value wins).

## 6. Copy the trained model back (on the Mac)

When the log ends with `End of training`:

```
mkdir -p outputs/min_smolvla/checkpoints
rsync -aL gpu:bimanual/outputs/min_smolvla/checkpoints/last/pretrained_model outputs/min_smolvla/checkpoints/last/
scp gpu:bimanual/min_smolvla.log outputs/
```

Only `pretrained_model` is needed for evaluation (it holds the weights, the config and the pre- and post-processors with the camera rename). `-L` follows the `last` link to the final checkpoint.

## 7. Stop the machine

Stop or delete the instance in the provider's dashboard: most bill until it's deleted, and some bill for the disk after it's stopped.

## 8. Evaluate (on the Mac)

Time one rollout first. SmolVLA predicts 50 actions per inference, so it should be quick on the Mac's GPU, but it hasn't been measured. If MPS fails, add `--device cpu`, which is slower (~4 min per rollout earlier):

```
python -m sim.evaluate outputs/min_smolvla/checkpoints/last/pretrained_model --variants S1:A --episodes 1
```

Then the three conditions, 50 rollouts per variant (as step 3, without `--temporal-ensemble`, which is ACT only):

```
V=S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right
C=outputs/min_smolvla/checkpoints/last/pretrained_model
for cond in train eval_colours eval_wording; do
  caffeinate -i -s python -m sim.evaluate $C --variants $V --episodes 50 --videos 2 --condition $cond --out outputs/min_smolvla/eval
done
```

Results go to `outputs/min_smolvla/eval/<condition>/summary.json`. Compare the train condition with step 3's ACT (`outputs/min_act/eval_100000_te/train/summary.json`): the gap is the cost of reading the command in language instead of a one-hot input (1tasks_scene.md, minimal plan step 4).

## ACT on the same machine

ACT trains on the one-hot view. It links `data/min`'s videos, so copy with `-L`:

```
rsync -aL --exclude sim data/min_onehot gpu:bimanual/data/     # on the Mac
bash train.sh act data/min_onehot outputs/min_act_long --steps=300000 2>&1 | tee min_act_long.log   # on the GPU, e.g. 3x longer to test undertraining (3train_evaluation.md)
```

Evaluate it with `--temporal-ensemble 0.01`, as every ACT result so far.

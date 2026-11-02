#!/bin/zsh
# Minimal plan step 3a: SmolVLA's evaluation. Waits for the run on the rented GPU to save its last checkpoint
# (020000), copies the model back (pretrained_model only, ~1.2 GB), waits for the step-3 colour evaluation to free
# the Mac's GPU, then evaluates the three conditions, 6 variants x 50 each. Run detached:
#   nohup caffeinate -i -s zsh scripts/eval_smolvla.sh > outputs/eval_smolvla.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
remote=bimanual/outputs/min_smolvla/checkpoints/020000/pretrained_model
ckpt=outputs/min_smolvla/checkpoints/020000/pretrained_model
until ssh -o ConnectTimeout=20 -o BatchMode=yes gpu "test -f $remote/model.safetensors && ! tmux has-session -t train 2>/dev/null" 2>/dev/null; do
  sleep 120
done
mkdir -p $ckpt
for i in 1 2 3; do
  rsync -a --partial gpu:$remote/ $ckpt/ && break
  echo "$(date '+%F %T') copy attempt $i failed"; sleep 60
done
[ -f $ckpt/model.safetensors ] || { echo "$(date '+%F %T') copy failed"; exit 1; }
echo "$(date '+%F %T') copied $ckpt"
while pgrep -f "eval_colours_step3.sh" >/dev/null; do sleep 60; done
for c in train eval_colours eval_wording; do
  echo "$(date '+%F %T') $c"
  .venv/bin/python -m sim.evaluate $ckpt --variants S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right \
    --condition $c --episodes 50 --videos 2 --out outputs/min_smolvla/eval > outputs/min_smolvla_$c.log 2>&1 \
    || echo "$(date '+%F %T') $c failed"
done
echo "$(date '+%F %T') done"

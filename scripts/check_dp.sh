#!/bin/zsh
# Early check on DP's training (minimal plan step 3b), as step 1 checked ACT at 40k: waits for checkpoint STEP on
# the rented GPU, copies the model back, runs sim.offline_check (normalisation floor, replay of training episodes),
# then a few closed-loop rollouts. Everything runs on the CPU, so it can run beside the Mac's GPU evaluation.
# 5000 steps at batch 64 is ~1.2 epochs, about where ACT's 40k check was. Run detached:
#   nohup caffeinate -i -s zsh scripts/check_dp.sh 005000 > outputs/check_dp_005000.out 2>&1 &
#   zsh scripts/check_dp.sh 010000 5     # a later checkpoint, 5 rollouts per variant
set -u
cd "$(dirname "$0")/.."
step=${1:-005000} n=${2:-3}
remote=bimanual/outputs/min_dp/checkpoints/$step/pretrained_model
ckpt=outputs/min_dp/checkpoints/$step/pretrained_model
until ssh -o ConnectTimeout=20 -o BatchMode=yes gpu "test -f $remote/model.safetensors" 2>/dev/null; do sleep 120; done
sleep 30  # let the save finish writing the processors
mkdir -p $ckpt
for i in 1 2 3; do
  rsync -a --partial gpu:$remote/ $ckpt/ && break
  echo "$(date '+%F %T') copy attempt $i failed"; sleep 60
done
[ -f $ckpt/model.safetensors ] || { echo "$(date '+%F %T') copy failed"; exit 1; }
echo "$(date '+%F %T') copied $ckpt"
.venv/bin/python -m sim.offline_check $ckpt data/min_onehot 2>&1 | grep -v "torchcodec\|libtorchcodec\|^Traceback\|^  File\|^    \|^OSError\|^\[end of"
echo "$(date '+%F %T') closed loop, $n rollouts per variant (CPU)"
.venv/bin/python -m sim.evaluate $ckpt --variants S1:A,T1:A:left --episodes $n --videos $n --device cpu \
  --out outputs/min_dp/eval_$step 2>&1 | grep "seed\|S1:\|T1:"
echo "$(date '+%F %T') done"

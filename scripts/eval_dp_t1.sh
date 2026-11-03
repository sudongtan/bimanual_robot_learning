#!/bin/zsh
# DP's T1 variants under the train and eval_colours conditions, 50 rollouts each, with 10 DDIM steps as
# scripts/eval_dp.sh. eval_dp.sh ran those two conditions without --variants, so sim.evaluate took all 20 variants:
# S1:A and S1:B were evaluated, then it ran S2:A (not in the minimal plan) and stopped on an error in S2's check
# tally, before any T1. Its S1 results stay in eval_100000_ddim10; T1 goes to its own directory so they aren't
# overwritten. Waits for the queue on the Mac's GPU. Run detached:
#   nohup caffeinate -i -s zsh scripts/eval_dp_t1.sh > outputs/eval_dp_t1.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/[e]val_dp.sh|scripts/[r]erun_start_pose.sh" >/dev/null; do sleep 60; done
ck=outputs/min_dp/checkpoints/100000/pretrained_model
for c in train eval_colours; do
  echo "$(date '+%F %T') DP $c, T1"
  .venv/bin/python -m sim.evaluate $ck --ddim 10 --variants T1:A:left,T1:A:right,T1:B:left,T1:B:right --condition $c \
    --videos 2 --out outputs/min_dp/eval_100000_ddim10_t1 > outputs/min_dp_${c}_t1.log 2>&1 || echo "$(date '+%F %T') DP $c T1 failed"
done
echo "$(date '+%F %T') done"

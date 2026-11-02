#!/bin/zsh
# Step 3's held-out colours alone (the second half of eval_colours.sh, restarted on its own). Run detached:
#   nohup caffeinate -i -s zsh scripts/eval_colours_step3.sh > outputs/eval_colours_step3.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
echo "$(date '+%F %T') step 3 (6 variants)"
.venv/bin/python -m sim.evaluate outputs/min_act/checkpoints/100000/pretrained_model \
  --variants S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right \
  --condition eval_colours --episodes 50 --videos 2 --temporal-ensemble 0.01 --out outputs/min_act/eval_100000_te \
  > outputs/min_eval_colours.log 2>&1 || echo "$(date '+%F %T') step 3 failed"
echo "$(date '+%F %T') done"

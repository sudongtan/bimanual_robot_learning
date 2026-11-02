#!/bin/zsh
# Held-out cube colours (--condition eval_colours) for the step-1 and step-3 ACT policies, as their train-condition
# results (100k checkpoints, 50 rollouts per variant, temporal ensembling). Run: caffeinate -i -s zsh scripts/eval_colours.sh
set -u
cd "$(dirname "$0")/.."
step() { echo "$(date '+%F %T') $*"; }
step "step 1 (S1:A)"
.venv/bin/python -m sim.evaluate outputs/s1a_act_action/checkpoints/100000/pretrained_model --variants S1:A \
  --condition eval_colours --episodes 50 --videos 2 --temporal-ensemble 0.01 --out outputs/s1a_act_action/eval_100000_te \
  > outputs/s1a_eval_colours.log 2>&1 || step "step 1 failed"
step "step 3 (6 variants)"
.venv/bin/python -m sim.evaluate outputs/min_act/checkpoints/100000/pretrained_model \
  --variants S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right \
  --condition eval_colours --episodes 50 --videos 2 --temporal-ensemble 0.01 --out outputs/min_act/eval_100000_te \
  > outputs/min_eval_colours.log 2>&1 || step "step 3 failed"
step done

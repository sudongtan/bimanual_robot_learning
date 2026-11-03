#!/bin/zsh
# Finding 11: rerun stress_start_pose for ACT and SmolVLA now that rollouts log the idle arm's distance outside the
# range between its start and its usual pose (`idle_arm_rad.away`), so moving it back home can be told from acting.
# Same starts and settings as scripts/stress.sh; new output directories, so the first runs stay as they were.
# DP's own stress_start_pose run (scripts/eval_dp.sh) already logs it. Waits for DP's evaluation. Run detached:
#   nohup caffeinate -i -s zsh scripts/rerun_start_pose.sh > outputs/rerun_start_pose.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/[e]val_dp.sh" >/dev/null; do sleep 60; done
V=S1:A,T1:A:left c=stress_start_pose
echo "$(date '+%F %T') ACT $c"
.venv/bin/python -m sim.evaluate outputs/min_act/checkpoints/100000/pretrained_model --variants $V --episodes 20 \
  --temporal-ensemble 0.01 --condition $c --out outputs/min_act/eval_100000_te_rerun > outputs/min_act_${c}_rerun.log 2>&1 \
  || echo "$(date '+%F %T') ACT $c failed"
echo "$(date '+%F %T') SmolVLA $c"
.venv/bin/python -m sim.evaluate outputs/min_smolvla/checkpoints/020000/pretrained_model --variants $V --episodes 20 \
  --condition $c --out outputs/min_smolvla/eval_rerun > outputs/min_smolvla_${c}_rerun.log 2>&1 \
  || echo "$(date '+%F %T') SmolVLA $c failed"
echo "$(date '+%F %T') done"

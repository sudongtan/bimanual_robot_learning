#!/bin/zsh
# Minimal plan step 5: stress tests (sim/evaluate.py, stress_* conditions), one factor at a time, on S1:A and
# T1:A:left, 20 rollouts each. Step 3's ACT (temporal ensembling) and SmolVLA; DP is added once it is trained.
# Waits for the SmolVLA evaluation to free the Mac's GPU. Run detached:
#   nohup caffeinate -i -s zsh scripts/stress.sh > outputs/stress.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
while pgrep -f "eval_colours_step3.sh|eval_smolvla.sh" >/dev/null; do sleep 60; done
V=S1:A,T1:A:left
for c in stress_distractors stress_start_pose stress_cube_yaw; do
  echo "$(date '+%F %T') ACT $c"
  .venv/bin/python -m sim.evaluate outputs/min_act/checkpoints/100000/pretrained_model --variants $V --episodes 20 \
    --videos 2 --temporal-ensemble 0.01 --condition $c --out outputs/min_act/eval_100000_te > outputs/min_act_$c.log 2>&1 \
    || echo "$(date '+%F %T') ACT $c failed"
  echo "$(date '+%F %T') SmolVLA $c"
  .venv/bin/python -m sim.evaluate outputs/min_smolvla/checkpoints/020000/pretrained_model --variants $V --episodes 20 \
    --videos 2 --condition $c --out outputs/min_smolvla/eval > outputs/min_smolvla_$c.log 2>&1 \
    || echo "$(date '+%F %T') SmolVLA $c failed"
done
echo "$(date '+%F %T') done"

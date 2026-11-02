#!/bin/zsh
# Minimal plan step 5a: the base policy alone (zero residual) on the 50 eval starts per variant, planning every 25
# frames (the residual's setting) and every 100 (LeRobot's default for ACT). Waits for the step-3 colour evaluation
# to free the GPU. Run: nohup caffeinate -i -s zsh scripts/residual_baseline.sh > outputs/residual_baseline.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
while pgrep -f "eval_colours_step3.sh" >/dev/null; do sleep 60; done
V=S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right
for k in 25 100; do
  echo "$(date '+%F %T') plan every $k"
  .venv/bin/python -m sim.residual outputs/min_act/checkpoints/100000/pretrained_model --variants $V --episodes 50 \
    --plan-every $k --out outputs/min_act/residual_base_k$k.json > outputs/residual_base_k$k.log 2>&1 || echo "$(date '+%F %T') failed"
done
echo "$(date '+%F %T') done"

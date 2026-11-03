#!/bin/zsh
# Minimal plan step 3b: Diffusion Policy's evaluation, as step 3's ACT (train and held-out colours, 6 variants x 50),
# then its stress tests (step 5, as scripts/stress.sh) and a check that DDIM costs nothing.
# DP is run with 10 DDIM steps instead of the 100 DDPM steps it was trained with (--ddim 10; Chi et al. 2023 run
# their real-robot policies this way): ~10x faster, and the offline replay error is the same (3train_evaluation.md).
# The check: the first 10 starts of S1:A and T1:A:left with the 100 DDPM steps, against the same starts with DDIM.
# Waits for the SmolVLA evaluation and the stress tests to free the Mac's GPU. Run detached:
#   nohup caffeinate -i -s zsh scripts/eval_dp.sh > outputs/eval_dp.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/[e]val_smolvla.sh|scripts/[s]tress.sh" >/dev/null; do sleep 60; done
ck=outputs/min_dp/checkpoints/100000/pretrained_model
for c in train eval_colours; do
  echo "$(date '+%F %T') DP $c"
  .venv/bin/python -m sim.evaluate $ck --ddim 10 --condition $c --videos 2 --out outputs/min_dp/eval_100000_ddim10 \
    > outputs/min_dp_$c.log 2>&1 || echo "$(date '+%F %T') DP $c failed"
done
for c in stress_distractors stress_start_pose stress_cube_yaw; do
  echo "$(date '+%F %T') DP $c"
  .venv/bin/python -m sim.evaluate $ck --ddim 10 --variants S1:A,T1:A:left --episodes 20 --condition $c --videos 2 \
    --out outputs/min_dp/eval_100000_ddim10 > outputs/min_dp_$c.log 2>&1 || echo "$(date '+%F %T') DP $c failed"
done
echo "$(date '+%F %T') DP train, 100 DDPM steps (DDIM check)"
.venv/bin/python -m sim.evaluate $ck --variants S1:A,T1:A:left --episodes 10 --videos 0 \
  --out outputs/min_dp/eval_100000_ddpm > outputs/min_dp_ddpm_check.log 2>&1 || echo "$(date '+%F %T') DP DDPM check failed"
echo "$(date '+%F %T') done"

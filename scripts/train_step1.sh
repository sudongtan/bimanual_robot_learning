#!/bin/zsh
# Plan step 1: ACT on S1 arm A, trained on `action`, then on `action_nominal` (sequential: one GPU).
# Restarted 2026-10-29 with STD_FLOOR in the stats (the first run without it: outputs/s1a_act_action_nofloor).
cd "$(dirname "$0")/.."
for label in action nominal; do
  root=data/s1a; [[ $label == nominal ]] && root=data/s1a_nominal
  .venv/bin/lerobot-train --dataset.repo_id=local/bimanual_so101 --dataset.root=$root \
    --policy.type=act --policy.device=mps --policy.push_to_hub=false \
    --output_dir=outputs/s1a_act_$label --job_name=s1a_act_$label \
    --steps=100000 --batch_size=8 --save_freq=20000 --log_freq=500 --wandb.enable=false --num_workers=4 \
    > outputs/s1a_act_$label.log 2>&1
  echo "$label exit $?"
done

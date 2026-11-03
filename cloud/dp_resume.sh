#!/usr/bin/env bash
# Resumes DP from its last checkpoint after the disk filled at step 55k (each checkpoint is 3.3 GB, 2.2 of it optimizer
# state). A pruner keeps the optimizer state of the latest complete checkpoint only; every checkpoint keeps its model.
cd "$(dirname "$0")"
c=outputs/min_dp/checkpoints
prune() {
  last=$(readlink $c/last)
  for d in $c/[0-9]*; do
    s=$(basename $d)
    [ "$((10#$s))" -lt "$((10#$last))" ] && rm -rf $d/training_state
  done
}
prune
( while true; do sleep 300; prune; done ) &
pruner=$!
source .venv-train/bin/activate
echo "$(date) DP resume from $(readlink $c/last); $(df -h / | tail -1)"
lerobot-train --config_path=$c/last/pretrained_model/train_config.json --resume=true 2>&1 | tee -a min_dp.log
echo "$(date) DP training ended"
kill $pruner; prune

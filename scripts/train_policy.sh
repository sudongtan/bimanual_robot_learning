#!/usr/bin/env bash
# Train one policy on one task's LeRobot dataset with LeRobot's lerobot-train.
#
#   scripts/train_policy.sh <task> <policy> [steps] [batch] [device]
#
#   <task>    a task in sim/waste_env.py TASKS, e.g. t0_can_to_metal_bin
#             (dataset: data/lerobot/<task>, from scripts/render_dataset.py)
#   <policy>  bc_mlp (our BC baseline, design-decisions D29), act (LeRobot's ACT) or
#             smolvla (LeRobot's SmolVLA, fine-tuned from lerobot/smolvla_base, D31)
#   steps     default 20000; batch default 8; device default mps (D33)
#
# Writes outputs/train/<task>/<policy>/ (checkpoints every 5000 steps, `last/` at the end)
# and the log outputs/train/<task>/<policy>.log. Refuses to start if the output folder
# exists, so a finished run is never overwritten; continue a run with LeRobot's
# `--config_path=<output>/checkpoints/last/pretrained_model/train_config.json --resume=true`.
#
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"; exit 1
fi
task=$1; policy=$2; steps=${3:-20000}; batch=${4:-8}; device=${5:-mps}
root=data/lerobot/$task
out=outputs/train/$task/$policy

cd "$(dirname "$0")/.."
[ -d "$root" ] || { echo "no dataset at $root (run scripts/render_dataset.py first)"; exit 1; }
[ -e "$out" ] && { echo "$out exists - not overwriting; resume it or choose another folder"; exit 1; }
mkdir -p "$(dirname "$out")"

echo "training $policy on $task: $steps steps, batch $batch, $device -> $out (log: $out.log)"
if [ "$policy" = smolvla ]; then
  # start from the pretrained model; it names its cameras camera1-3 (debug-log #34)
  model=(--policy.path=lerobot/smolvla_base
         --rename_map='{"observation.images.overview": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}')
else
  model=(--policy.type=$policy)
fi
uv run python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/$task --dataset.root=$root \
  "${model[@]}" --policy.push_to_hub=false --policy.device=$device \
  --steps=$steps --batch_size=$batch --save_freq=5000 --log_freq=500 --env_eval_freq=0 \
  --output_dir=$out > "$out.log" 2>&1
echo "done: $out/checkpoints/last/pretrained_model"

#!/usr/bin/env bash
# Train on a rented CUDA GPU (1tasks_scene.md, "Minimal plan"). Training needs LeRobot only, not MuJoCo;
# evaluation stays on the Mac (python -m sim.evaluate CKPT ...).
#
#   Mac:  rsync -aL data/min_view HOST:bimanual/data/      # -L: a dataset view links its videos, copy the files
#         rsync -a cloud/train.sh HOST:bimanual/
#   GPU:  cd bimanual && bash train.sh smolvla data/min_view outputs/min_smolvla
#         bash train.sh act data/min_view outputs/min_act          # ACT with the variant input (view made with --variant-input)
#   Mac:  rsync -a HOST:bimanual/outputs/min_smolvla/checkpoints/last outputs/min_smolvla/checkpoints/
#
# Extra arguments go to lerobot-train (e.g. --batch_size=32 if SmolVLA runs out of memory on a 24 GB card).
set -euo pipefail
policy=$1 root=$2 out=$3
shift 3
pip install -q "lerobot[smolvla,training,dataset]==0.6.1"
common=(--dataset.repo_id=local/bimanual_so101 --dataset.root="$root" --policy.device=cuda --policy.push_to_hub=false
        --output_dir="$out" --job_name="$(basename "$out")" --wandb.enable=false --num_workers=8 --save_freq=5000)
case $policy in
  # ACT: LeRobot's ACT setting (as plan step 1 on the Mac).
  act) lerobot-train --policy.type=act --steps=100000 --batch_size=8 "${common[@]}" "$@" ;;
  # SmolVLA: LeRobot's fine-tuning recipe from the pretrained base (batch 64, 20k steps). The base model names its
  # cameras camera1-3; the rename map maps ours onto them and is saved with the preprocessor for evaluation.
  smolvla) lerobot-train --policy.path=lerobot/smolvla_base --steps=20000 --batch_size=64 \
             --rename_map='{"observation.images.overview": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}' \
             "${common[@]}" "$@" ;;
  *) echo "policy: act or smolvla" >&2; exit 1 ;;
esac

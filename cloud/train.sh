#!/usr/bin/env bash
# Train on a rented CUDA GPU (1tasks_scene.md, "Minimal plan"). Training needs LeRobot only, not MuJoCo;
# evaluation stays on the Mac (python -m sim.evaluate CKPT ...).
#
#   Mac:  rsync -aL --exclude sim data/min HOST:bimanual/data/     # the LeRobot dataset (~2.5 GB; sim/ is the recorder's states, not needed)
#         rsync -aL --exclude sim data/min_onehot HOST:bimanual/data/   # ACT only: the view links data/min's videos, -L copies them
#         rsync -a cloud/train.sh HOST:bimanual/
#   GPU:  cd bimanual && bash train.sh smolvla data/min outputs/min_smolvla      # SmolVLA reads each episode's command
#         bash train.sh act data/min_onehot outputs/min_act                     # ACT with the one-hot variant input
#   Mac:  rsync -a HOST:bimanual/outputs/min_smolvla/checkpoints/last outputs/min_smolvla/checkpoints/
#
# Extra arguments go to lerobot-train and override the defaults below (the last value wins), e.g.
# --batch_size=32 --steps=40000 if SmolVLA runs out of memory on a 24 GB card (same number of samples).
# LeRobot 0.6.1 needs Python >= 3.12, which many GPU images don't have, so it is installed with uv into .venv-train.
set -euo pipefail
policy=$1 root=$2 out=$3
shift 3
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv-train ] || uv venv -q -p 3.12 .venv-train
source .venv-train/bin/activate
uv pip install -q "lerobot[smolvla,training,dataset]==0.6.1"
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

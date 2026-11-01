#!/bin/zsh
# Minimal plan steps 2-3 (1tasks_scene.md), unattended (run: caffeinate -i -s zsh scripts/step2_3.sh): record the 6 variants (50 each; S1:A repeats data/s1a's
# seeds, so its episodes are identical), check them, add the one-hot variant input, train ACT, evaluate.
# Imbalance: no reweighting in this baseline (3train_evaluation.md, finding 5).
set -u
cd "$(dirname "$0")/.."
V=S1:A,S1:B,T1:A:left,T1:A:right,T1:B:left,T1:B:right
step() { echo "$(date '+%F %T') $*"; }
step record;   .venv/bin/python -m sim.record data/min --variants $V --per-variant 50 > outputs/min_record.log 2>&1 || { step "record failed"; exit 1; }
step check;    .venv/bin/python -m sim.check_data data/min > outputs/min_check.log 2>&1 || step "check failed (continuing)"
step view;     .venv/bin/python -m sim.dataset_view data/min data/min_onehot --variant-input > outputs/min_view.log 2>&1 || { step "view failed"; exit 1; }
step train
.venv/bin/lerobot-train --dataset.repo_id=local/bimanual_so101 --dataset.root=data/min_onehot \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --output_dir=outputs/min_act --job_name=min_act \
  --steps=100000 --batch_size=8 --save_freq=20000 --log_freq=500 --wandb.enable=false --num_workers=4 \
  > outputs/min_act.log 2>&1 || { step "train failed"; exit 1; }
step evaluate
.venv/bin/python -m sim.evaluate outputs/min_act/checkpoints/100000/pretrained_model --variants $V --episodes 50 \
  --videos 2 --temporal-ensemble 0.01 --out outputs/min_act/eval_100000_te > outputs/min_eval.log 2>&1 || step "evaluate failed"
step done

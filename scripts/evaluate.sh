#!/usr/bin/env bash
# Evaluate one policy on one task: run it on the evaluation seeds and count successes.
#
#   scripts/evaluate.sh <task> <policy> [first_seed] [n_episodes] [device]
#
#   <task>      a task in sim/waste_env.py TASKS, e.g. t0_can_to_metal_bin
#   <policy>    expert                -> the scripted expert (scripts/evaluate_expert.py)
#               bc_mlp | act | smolvla -> the trained checkpoint
#                                        outputs/train/<task>/<policy>/checkpoints/last/pretrained_model,
#                                        run by LeRobot's lerobot-eval in our environment
#   first_seed, n_episodes  default 1000 and 50: THE evaluation set, seeds 1000-1049
#                           (design-decisions D27) — the same for every policy
#   device      for trained policies, default mps
#
# Writes outputs/eval/<task>/<policy>/ (a different seed range: <policy>_seeds<first>-<last>/):
#   eval_info.json   the evaluator's full results (LeRobot's layout, or the expert script's)
#   summary.json     the same success rate in one layout for every policy
# and the log outputs/eval/<task>/<policy>.log. Refuses to overwrite an existing result.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,19p' "$0"; exit 1
fi
task=$1; policy=$2; first=${3:-1000}; n=${4:-50}; device=${5:-mps}
last=$((first + n - 1))
name=$policy; [ "$first" = 1000 ] && [ "$n" = 50 ] || name=${policy}_seeds${first}-${last}
out=outputs/eval/$task/$name

cd "$(dirname "$0")/.."
[ -e "$out" ] && { echo "$out exists - not overwriting"; exit 1; }
mkdir -p "$(dirname "$out")"

if [ "$policy" = expert ]; then
  echo "evaluating $policy on $task, seeds $first-$last -> $out (log: $out.log)"
  uv run python scripts/evaluate_expert.py --task "$task" --seed "$first" --n-episodes "$n" \
    --out "$out" > "$out.log" 2>&1
else
  ckpt=outputs/train/$task/$policy/checkpoints/last/pretrained_model
  [ -d "$ckpt" ] || { echo "no trained checkpoint at $ckpt (train it first: scripts/train_policy.sh $task $policy)"; exit 1; }
  # `last/` already exists mid-training (it points at the newest intermediate checkpoint),
  # so require LeRobot's end-of-training line in the training log
  grep -q "End of training" "outputs/train/$task/$policy.log" 2>/dev/null \
    || { echo "training of $policy on $task has not finished (outputs/train/$task/$policy.log)"; exit 1; }
  echo "evaluating $policy on $task, seeds $first-$last -> $out (log: $out.log)"
  extra=()
  if [ "$policy" = smolvla ]; then  # the pretrained SmolVLA names its cameras camera1-3 (debug-log #34)
    extra=(--rename_map='{"observation.images.overview": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}')
  fi
  uv run python -m lerobot.scripts.lerobot_eval \
    --policy.path="$ckpt" --policy.device="$device" \
    --env.type=waste_sort --env.discover_packages_path=sim.lerobot_plugin --env.task="$task" \
    --eval.n_episodes="$n" --eval.batch_size=1 --seed="$first" \
    --output_dir="$out" "${extra[@]+"${extra[@]}"}" > "$out.log" 2>&1
fi

# one layout for every policy: summary.json
uv run python - "$out" "$task" "$policy" "$first" "$n" <<'EOF'
import json, sys
from pathlib import Path
out, task, policy, first, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
info = json.loads((out / "eval_info.json").read_text())
agg = info.get("overall") or info["aggregated"]          # lerobot-eval | evaluate_expert.py
s = {"task": task, "policy": policy, "seeds": [first, first + n - 1], "n_episodes": n,
     "pc_success": agg["pc_success"], "n_success": round(agg["pc_success"] * n / 100),
     "eval_s": agg.get("eval_s")}
(out / "summary.json").write_text(json.dumps(s, indent=2))
print(f"{policy} on {task}: {s['n_success']}/{n} successful ({s['pc_success']:.0f}%) -> {out}/summary.json")
EOF

#!/usr/bin/env bash
# Quick check on the rented GPU before a real run (docs/4cloud_training.md, step 6): 30 training steps through train.sh, at
# the policy's batch and, if that runs out of memory, at half of it. Exercises everything the real run needs: the
# install, the base-model download, video decoding, the GPU, memory and saving a checkpoint. Prints which batch to
# use and the full run's time, and writes the real run's command to smoke_next_<policy>.sh. ~5-10 min the first time
# (it includes the install). Leaves nothing behind but logs.
#
#   bash smoke_test.sh [POLICY] [DATASET] [OUT]   # defaults: smolvla data/min outputs/min_<policy>
#   bash smoke_test.sh dp data/min_onehot          # Diffusion Policy needs the one-hot view
set -uo pipefail
cd "$(dirname "$0")"
policy=${1:-smolvla} root=${2:-data/min} final=${3:-outputs/min_$policy}
case $policy in  # train.sh's steps and batch for each policy
  smolvla) steps=20000 batches="64 32" ;;
  dp) steps=100000 batches="64 32" ;;
  act) steps=100000 batches="8" ;;
  *) echo "FAIL: policy must be smolvla, dp or act"; exit 1 ;;
esac
base=${batches%% *}

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "FAIL: no NVIDIA GPU visible"; exit 1; }
[ -f "$root/meta/info.json" ] || { echo "FAIL: no dataset at $root (copy it first: docs/4cloud_training.md, step 5)"; exit 1; }
echo "dataset: $root, $(grep -o '"total_episodes": [0-9]*' "$root/meta/info.json")"

for bs in $batches; do
  out=outputs/smoke_${policy}_b$bs log=smoke_${policy}_b$bs.log
  rm -rf "$out"
  echo "== $policy, batch $bs, 30 steps (log: $log)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > smoke_mem.log 2>/dev/null &
  mon=$!
  bash train.sh "$policy" "$root" "$out" --batch_size=$bs --steps=30 --log_freq=10 --save_freq=30 > "$log" 2>&1
  kill $mon 2>/dev/null
  peak=$(sort -n smoke_mem.log | tail -1)
  if grep -q "End of training" "$log"; then
    t=$(grep -o "updt_s:[0-9.]*" "$log" | tail -1 | cut -d: -f2)
    d=$(grep -o "data_s:[0-9.]*" "$log" | tail -1 | cut -d: -f2)
    loss=$(grep -o "loss:[0-9.]*" "$log" | sed -n '1p;$p' | cut -d: -f2 | paste -sd' ' - | sed 's/ / -> /')
    n=$((steps * base / bs))
    ckpt=$(ls "$out"/checkpoints/*/pretrained_model/model.safetensors 2>/dev/null | head -1)
    echo "OK at batch $bs: ${t} s/step (data loading ${d} s), loss ${loss}, peak GPU memory ${peak} MiB"
    [ -n "$ckpt" ] && echo "checkpoint saved: $ckpt" || echo "WARNING: no checkpoint found under $out/checkpoints"
    awk -v t="$t" -v d="$d" -v n="$n" 'BEGIN {
      printf "full run: %d steps x %.2f s = ~%.1f h\n", n, t + d, n * (t + d) / 3600
      if (d > 0.3 * t) print "WARNING: data loading is slow next to the GPU step; a machine with more CPU cores would help" }'
    extra=""
    [ "$bs" != "$base" ] && extra=" --batch_size=$bs --steps=$n"
    next="bash train.sh $policy $root $final$extra 2>&1 | tee $(basename "$final").log"
    echo "$next" > "smoke_next_$policy.sh"
    echo "NEXT: $next"
    rm -rf "$out" smoke_mem.log
    exit 0
  elif grep -qi "out of memory" "$log"; then
    echo "batch $bs: out of memory (peak ${peak} MiB)"
    rm -rf "$out"
  else
    echo "FAIL at batch $bs; last lines of $log:"
    grep -v "^ \|torchcodec\|FFmpeg" "$log" | tail -15
    exit 1
  fi
done
echo "FAIL: out of memory at every batch tried ($batches); try a smaller --batch_size, or a GPU with more memory"
exit 1

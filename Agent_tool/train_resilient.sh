#!/usr/bin/env bash
# Crash-resilient training: run train_ppo.py, and if it dies (the rare MuJoCo/numpy
# segfault on long runs), resume from the latest checkpoint until it reaches --timesteps.
# Single-threads each process (reduces the corruption) and retries up to MAX_TRIES.
#
# Usage (from repo root):
#   Agent_tool/train_resilient.sh <run-name> <timesteps> <n-asteroids> [extra train_ppo.py args...]
# Example:
#   Agent_tool/train_resilient.sh ppo_rebuild_v5 3000000 90 --curriculum --n-start 10
set -u
cd "$(dirname "$0")/.."

RUN="${1:?run-name}"; STEPS="${2:?timesteps}"; NAST="${3:?n-asteroids}"; shift 3
EXTRA="$*"
OUT="logs/$RUN/train.out"
MAX_TRIES=12
mkdir -p "logs/$RUN"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1

for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== attempt $i ($(date -u +%H:%M:%S)) ===" >> "$OUT"
  conda run -n asteroid-belt-runner python train/train_ppo.py \
    --timesteps "$STEPS" --n-envs 16 --n-asteroids "$NAST" \
    --run-name "$RUN" --resume $EXTRA >> "$OUT" 2>&1
  if grep -q "saved final model" "$OUT"; then
    echo "TRAIN_COMPLETE after $i attempt(s)"; exit 0
  fi
  echo "attempt $i ended without completion (likely crash); resuming..."
done
echo "TRAIN_GAVE_UP after $MAX_TRIES attempts"; exit 1

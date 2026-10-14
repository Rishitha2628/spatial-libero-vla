#!/bin/bash
# Train SmolVLA on one dataset, then evaluate it on LIBERO-Spatial (normal + swapped
# instructions). Every condition gets the same budget: 20k steps, batch 8.
#
#   bash scripts/run_condition.sh mg_200
#   -> outputs/smolvla_mg_200/final, results/mg_200.json, results/mg_200_swapped.json
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/env.sh" >/dev/null
NAME="$1"; STEPS="${STEPS:-20000}"; EPISODES="${EPISODES:-50}"
MODEL="$HERE/outputs/smolvla_$NAME/final"
mkdir -p "$HERE/logs" "$HERE/results"

if [ ! -f "$MODEL/model.safetensors" ]; then
  python "$HERE/scripts/train_smolvla.py" --dataset "$NAME" --steps "$STEPS" \
    --save-every 1000000 > "$HERE/logs/train_$NAME.log" 2>&1
fi
python "$HERE/scripts/eval_spatial.py" --model-dir "$MODEL" --episodes "$EPISODES" \
  --label "$NAME" --out "$HERE/results/$NAME.json" > "$HERE/logs/eval_$NAME.log" 2>&1
# SKIP_SWAP=1: the swap test gave 0% for every condition so far, so it no longer
# separates conditions and costs ~2 h of 600-step failures.
if [ -z "$SKIP_SWAP" ]; then
  python "$HERE/scripts/eval_spatial.py" --model-dir "$MODEL" --episodes "$EPISODES" --swap-instructions \
    --label "${NAME}_swapped" --out "$HERE/results/${NAME}_swapped.json" > "$HERE/logs/eval_${NAME}_swapped.log" 2>&1
fi
grep OVERALL "$HERE"/logs/eval_${NAME}*.log

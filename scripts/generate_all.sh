#!/bin/bash
# Generate the MimicGen dataset for all ten LIBERO-Spatial tasks, $JOBS at a time.
#   bash scripts/generate_all.sh 200                                   # from data/source -> data/generated
#   SRC=data/source_10 OUT=data/generated_src10 TAG=src10 bash scripts/generate_all.sh 200
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/env.sh" >/dev/null
NUM="${1:-200}"; JOBS="${JOBS:-2}"   # each worker ~2.2 GB RAM; 3 froze the desktop once Chrome was open, 5 thrashes
SRC="$HERE/${SRC:-data/source}"; OUT="$HERE/${OUT:-data/generated}"; TAG="${TAG:+_$TAG}"
mkdir -p "$HERE/logs"
seq 0 9 | xargs -P "$JOBS" -I{} sh -c \
  "python '$HERE/scripts/generate.py' --task {} --num $NUM --source '$SRC' --out '$OUT' > '$HERE/logs/generate${TAG}_task{}.log' 2>&1"
cat "$OUT"/task*_stats.json

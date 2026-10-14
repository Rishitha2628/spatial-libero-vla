#!/bin/bash
# MimicGen from 10 source demos PER TASK (100 total) -> 200 generated/task -> train + eval.
# Same generation budget (200/task) and training budget as mg_200, so the only change
# vs mg_200 is 1 -> 10 source demos per task.
#
#   nohup bash scripts/run_mg10.sh > logs/run_mg10.log 2>&1 &
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/env.sh" >/dev/null
cd "$HERE"
NAME=mg10_200

need_gb() {  # stop cleanly rather than crash mid-step on a full disk
  local free; free=$(df --output=avail -BG "$HERE" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$1" ]; then
    echo "[$(date +%H:%M)] STOP: $2 needs ${1} GB free, only ${free} GB available"; exit 1
  fi
  echo "[$(date +%H:%M)] disk ok for $2: ${free} GB free (need $1)"
}

# 1. official LIBERO-Spatial demos (deleted earlier to save disk)
if [ ! -d data/source_10 ]; then
  need_gb 7 "download"
  # the network here drops out; snapshot_download then returns silently with nothing,
  # so retry until all ten files are really on disk
  for attempt in $(seq 1 20); do
    python - <<'PY' || true
from huggingface_hub import snapshot_download
snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset',
                  allow_patterns=['libero_spatial/*'], local_dir='data', max_workers=4)
PY
    n=$(ls data/libero_spatial/*_demo.hdf5 2>/dev/null | wc -l)
    [ "$n" -eq 10 ] && break
    echo "[$(date +%H:%M)] download attempt $attempt: $n/10 files, retrying in 60 s"; sleep 60
  done
  [ "$(ls data/libero_spatial/*_demo.hdf5 2>/dev/null | wc -l)" -eq 10 ] || { echo "STOP: download failed"; exit 1; }
  # 2. annotate the first 10 demos of each task, then drop the raw files
  python scripts/prepare_source.py --per-task 10 --out data/source_10
  rm -rf data/libero_spatial data/.cache
fi

# 3. generate 200/task from the 10-per-task pool
if [ ! -f data/generated_src10/task9_stats.json ] && [ ! -d data/lerobot/$NAME ]; then
  need_gb 9 "generation"
  SRC=data/source_10 OUT=data/generated_src10 TAG=src10 bash scripts/generate_all.sh 200
fi

# 4. convert (deleting each hdf5 as it is converted), fix image stats -> builds the arrow cache
if [ ! -f data/lerobot/$NAME/meta/stats.json.orig ]; then
  need_gb 12 "conversion"
  python scripts/convert_to_lerobot.py --src data/generated_src10 --name $NAME --threads 8 --delete-src
fi

# 5. train + eval (no swap test)
need_gb 2 "training checkpoint"
SKIP_SWAP=1 bash scripts/run_condition.sh $NAME
echo "[$(date +%H:%M)] DONE"

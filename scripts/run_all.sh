#!/bin/bash
# The three conditions, one after another (one GPU):
#   official_50 -- 50 human demos/task (500)       the standard LIBERO-Spatial setup
#   mg_200      -- 200 MimicGen demos/task (2000)  generated from 10 human demos
#   source_1    -- the same 10 human demos alone    what MimicGen started from
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$HERE/scripts/run_condition.sh" official_50
# mg_200 may still be converting when this starts
until grep -q "wrote .*mg_200/meta/stats.json" "$HERE/logs/convert_mg_200.log" 2>/dev/null; do sleep 60; done
bash "$HERE/scripts/run_condition.sh" mg_200
bash "$HERE/scripts/run_condition.sh" source_1

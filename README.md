# MimicGen for LIBERO-Spatial

Generate a large training set from a handful of human demos with
[MimicGen](https://github.com/NVlabs/mimicgen), train
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) on it, and evaluate on
[LIBERO-Spatial](https://github.com/Lifelong-Robot-Learning/LIBERO).

All 10 LIBERO-Spatial tasks share one goal (put the black bowl on the plate), so a
single MimicGen adapter with two subtasks (grasp relative to the bowl, place relative
to the plate) covers every task.

## Results

SmolVLA, 20k steps × batch 8, one seed per condition. Evaluated on the 50 fixed init
states × 10 tasks = 500 episodes.

| condition | training data | success |
|---|---|---|
| `official_50` | 500 human demos (50/task) | **61.6%** |
| `mg_200` | 2000 MimicGen demos from 10 human demos (1/task) | **54.4%** |
| `mg10_200` | 2000 MimicGen demos from 100 human demos (10/task) | **52.8%** |
| `source_1` | the 10 human demos alone | **46.0%** |

- MimicGen adds +8.4 pts over its 10 source demos, closing about half the gap to 500
  human demos. Mostly by fixing tasks where `source_1` collapses (tasks 4/6/8:
  2/0/12% → 54/50/54%).
- 10 source demos per task instead of 1 gives no gain (within the ~2 pt standard
  error). LIBERO-Spatial eval layouts barely vary (bowl xy std ≈ 0.8 cm), which
  limits what extra diversity can show.
- Swapping instructions between tasks drops every policy to 0%, so the policies do
  use the language.

Per-task numbers: `results/*.json`.

## Pipeline

```bash
source env.sh
python scripts/prepare_source.py --per-task 1 --out data/source   # pick + annotate source demos
bash   scripts/generate_all.sh 200                                # MimicGen → data/generated/*.hdf5
python scripts/convert_to_lerobot.py --src data/generated --name mg_200
bash   scripts/run_condition.sh mg_200                            # train + eval → results/mg_200.json
```

| file | role |
|---|---|
| `mg_libero.py` | MimicGen ↔ LIBERO adapter: env interface, env wrapper, task spec |
| `prepare_source.py` | replays official demos to add MimicGen annotations (poses, grasp signal) |
| `generate.py`, `generate_all.sh` | generate demos per task; keep only successes |
| `convert_to_lerobot.py`, `fix_image_stats.py` | build a LeRobot dataset; fix LeRobot's image-std bug |
| `train_smolvla.py`, `eval_spatial.py` | train SmolVLA; closed-loop eval (optionally with swapped instructions) |
| `run_all.sh`, `run_mg10.sh` | reproduce every condition above |

## Setup

Python 3.10, robosuite 1.4.1, mujoco 3.9.0, torch 2.6, lerobot 0.4.4, LIBERO at
`$LIBERO_DIR` (default `~/LIBERO`), mimicgen and robomimic installed editable from
`third_party/`. Install with `-c scripts/pip_constraints.txt` so numpy/torch/mujoco
stay pinned. Then `source env.sh && python scripts/check_env.py`.

Notes:
- Each generation process uses ~2.2 GB of RAM. `JOBS` (default 2) sets how many run
  in parallel.
- Loading a LeRobot dataset writes a full arrow cache under `$HF_HOME/datasets`, about
  as large as the dataset itself (~9 GB for 2000 demos).

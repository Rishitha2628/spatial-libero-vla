"""Convert LIBERO-format hdf5 files into one multi-task LeRobot dataset.

Same converter for every training condition, so the only thing that differs
between them is the data:

  generated   python scripts/convert_to_lerobot.py --src data/generated --name mg_200
  official    python scripts/convert_to_lerobot.py --src data/libero_spatial --name official_50
  source only python scripts/convert_to_lerobot.py --src data/libero_spatial --demos-per-task 1 --name source_1

The per-frame `task` string is the instruction, read from the file's own
problem_info, because it is fed to the VLM's tokenizer.

Ends by running fix_image_stats.py -- LeRobot's writer stores image std ~0.002
instead of ~0.16, which silently stops the vision tower learning.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 20          # LIBERO control frequency
IMG_SIZE = 128    # native LIBERO resolution
PROJ = Path(__file__).resolve().parent.parent


def instruction_of(f):
    return json.loads(f["data"].attrs["problem_info"])["language_instruction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="directory of *.hdf5 files")
    ap.add_argument("--name", required=True, help="dataset name under data/lerobot/")
    ap.add_argument("--demos-per-task", type=int, default=0, help="0 = all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--delete-src", action="store_true",
                    help="delete each hdf5 once converted (keeps the disk peak low)")
    args = ap.parse_args()

    files = sorted(p for p in args.src.glob("*.hdf5"))
    if not files:
        sys.exit(f"no hdf5 files under {args.src}")
    out = PROJ / "data" / "lerobot" / args.name
    if out.exists():
        shutil.rmtree(out)

    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.name}", fps=FPS, root=out,
        features={
            "observation.images.image":  {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3),
                                          "names": ["height", "width", "channel"]},
            "observation.images.image2": {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3),
                                          "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": (8,),
                                  "names": ["x", "y", "z", "ax", "ay", "az", "g0", "g1"]},
            "action": {"dtype": "float32", "shape": (7,),
                       "names": ["dx", "dy", "dz", "dr", "dp", "dyw", "gripper"]},
        },
        image_writer_threads=args.threads,
    )

    total_frames = total_eps = 0
    for path in files:
        with h5py.File(path, "r") as f:
            instruction = instruction_of(f)
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if args.demos_per_task:
                keys = keys[: args.demos_per_task]
            for k in keys:
                demo = f["data"][k]
                obs = demo["obs"]
                actions = demo["actions"][:]
                agent, wrist = obs["agentview_rgb"][:], obs["eye_in_hand_rgb"][:]
                state = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], 1)
                for t in range(len(actions)):
                    dataset.add_frame({
                        "observation.images.image": agent[t],
                        "observation.images.image2": wrist[t],
                        "observation.state": state[t].astype(np.float32),
                        "action": actions[t].astype(np.float32),
                        "task": instruction,
                    })
                dataset.save_episode()
                total_frames += len(actions)
                total_eps += 1
        print(f"[{total_eps:4d} eps {total_frames:7d} frames] {path.name}: {instruction}", flush=True)
        if args.delete_src:
            path.unlink()

    dataset.finalize()
    print(f"finalized {out}: {total_eps} episodes, {total_frames} frames", flush=True)
    subprocess.run([sys.executable, str(PROJ / "scripts" / "fix_image_stats.py"),
                    "--root", str(out), "--repo-id", f"local/{args.name}"], check=True)


if __name__ == "__main__":
    main()

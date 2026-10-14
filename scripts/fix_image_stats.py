"""Recompute correct per-pixel image statistics for a converted LeRobot dataset.

The stats LeRobot writes for our image columns are the std of each frame's
*mean* pixel value, not the std across pixels -- roughly 0.002 instead of 0.16
for the agentview camera. Since the policy normalizes images with MEAN_STD,
dividing by that tiny std pushes normalized pixels to about +/-300 and the
vision backbone learns nothing. This rewrites meta/stats.json with true
per-pixel mean/std.

  python scripts/fix_image_stats.py --root data/lerobot/<name>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch



def true_image_stats(dataset, keys, n_samples):
    """Per-channel mean/std over pixels, from an evenly spaced sample of frames."""
    idx = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset))).astype(int)
    out = {}
    for key in keys:
        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0
        for i in idx:
            img = dataset[int(i)][key].double()      # (3, H, W) in [0, 1]
            total += img.sum(dim=(1, 2))
            total_sq += (img ** 2).sum(dim=(1, 2))
            count += img.shape[1] * img.shape[2]
        mean = total / count
        std = torch.sqrt((total_sq / count - mean ** 2).clamp_min(0))
        out[key] = (mean.float().numpy(), std.float().numpy())
        print(f"  {key}: mean={np.round(mean.numpy(), 4)} std={np.round(std.numpy(), 4)} "
              f"({len(idx)} frames)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo-id", default=None, help="defaults to local/<root dir name>")
    ap.add_argument("--n-samples", type=int, default=1500)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.repo_id or f"local/{args.root.name}", root=args.root)
    image_keys = [k for k in dataset.meta.stats if k.startswith("observation.images.")]
    print(f"recomputing stats for {image_keys} over {len(dataset)} frames")

    stats = true_image_stats(dataset, image_keys, args.n_samples)

    stats_path = args.root / "meta" / "stats.json"
    on_disk = json.loads(stats_path.read_text())
    for key, (mean, std) in stats.items():
        old = np.array(on_disk[key]["std"], dtype=float)
        # Keep the stored nesting (stats are shaped (3, 1, 1) for images).
        shape = old.shape
        on_disk[key]["mean"] = np.asarray(mean).reshape(shape).tolist()
        on_disk[key]["std"] = np.asarray(std).reshape(shape).tolist()
        print(f"  {key}: std {old.ravel().round(4)} -> {np.asarray(std).ravel().round(4)}")

    backup = stats_path.with_suffix(".json.orig")
    if not backup.exists():
        backup.write_text(stats_path.read_text())
        print(f"backed up original to {backup}")
    stats_path.write_text(json.dumps(on_disk, indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()

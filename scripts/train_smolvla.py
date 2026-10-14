"""Fine-tune SmolVLA on a LIBERO-Spatial LeRobot dataset (language-conditioned).

  python scripts/train_smolvla.py --dataset mg_200 --steps 20000
  -> outputs/smolvla_mg_200/{checkpoint_*, final}

Carried over from vla-libero-project, where each of these was a silent 0%:
  * delta_timestamps, so `action` is the true next chunk rather than one action repeated;
  * normalization lives in the pre/post-processor pipelines, saved next to the weights;
  * the VLM is frozen (SmolVLA's defaults) -- only the action expert trains, which is
    what fits in 6 GB of VRAM.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

PROJ = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="name under data/lerobot/")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--base", default="lerobot/smolvla_base")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--img", type=int, default=256, help="VLM input resolution")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    root = PROJ / "data" / "lerobot" / args.dataset
    repo_id = f"local/{args.dataset}"
    out = args.output_dir or PROJ / "outputs" / f"smolvla_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    state_dim = meta.features["observation.state"]["shape"][0]
    action_dim = meta.features["action"]["shape"][0]
    print(f"{args.dataset}: fps={meta.fps} frames={meta.total_frames} "
          f"episodes={meta.total_episodes} tasks={len(meta.tasks)}", flush=True)

    cfg = SmolVLAConfig(
        input_features={
            "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state":         PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        resize_imgs_with_padding=(args.img, args.img),
        freeze_vision_encoder=True,
        train_expert_only=True,
        optimizer_lr=args.lr,
        device=args.device,
    )

    delta_timestamps = {"action": [i / meta.fps for i in cfg.action_delta_indices]}
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    policy = SmolVLAPolicy.from_pretrained(args.base, config=cfg).to(args.device)
    policy.train()
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params {trainable / 1e6:.1f}M", flush=True)
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    params = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, betas=cfg.optimizer_betas,
                              eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)

    def save(tag):
        d = out / tag
        policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)
        (d / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
        print(f"saved {d}", flush=True)

    log = open(out / "train_log.csv", "w")
    log.write("step,loss,lr,samples_per_s\n")
    step, losses, t0 = 0, [], time.time()
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = pre(batch)
        loss, _ = policy.forward(batch)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm)
        optim.step()
        sched.step()
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            avg = sum(losses[-100:]) / 100
            sps = 100 * args.batch_size / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step}/{args.steps} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {sps:.1f} samp/s | peak {mem:.2f}GiB", flush=True)
            log.write(f"{step},{avg:.6f},{sched.get_last_lr()[0]:.3e},{sps:.2f}\n")
            log.flush()
            t0 = time.time()
        if step % args.save_every == 0 and step < args.steps:
            save(f"checkpoint_{step}")
    save("final")
    log.close()


if __name__ == "__main__":
    main()

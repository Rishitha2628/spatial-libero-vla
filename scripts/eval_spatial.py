"""Closed-loop success rate on LIBERO-Spatial, on LIBERO's fixed eval init states.

  python scripts/eval_spatial.py --model-dir outputs/smolvla_mg_200/final --episodes 50
  python scripts/eval_spatial.py --model-dir ... --swap-instructions   # language test

Protocol (same as the previous project and LIBERO's own): reset, set one of the 50
fixed init states, 10 settle steps of zero action, then up to 600 policy steps;
success is LIBERO's goal predicate on(akita_black_bowl_1, plate_1).

--swap-instructions gives each task the NEXT task's instruction. In LIBERO-Spatial
the scene layout already differs per task, so a policy can partly succeed by
reading the layout; how far success drops under a swap measures how much the
instruction itself is being used.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from libero_common import NUM_TASKS, SETTLE_ACTION, SETTLE_STEPS, eval_init_states, make_env, task_info


def load_policy(model_dir, device="cuda"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, policy_action_to_transition,
        transition_to_batch, transition_to_policy_action,
    )
    model_dir = str(model_dir)
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.config.device = device
    policy.to(device).eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=model_dir, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=model_dir, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    for pipeline in (pre, post):
        for step in getattr(pipeline, "steps", []):
            if getattr(step, "device", None) not in (None, "cpu"):
                step.device = device
    return policy, pre, post


def build_obs(obs):
    """Same features the converter wrote: raw (opengl-convention) images, [eef pos, axis-angle, gripper]."""
    import robosuite.utils.transform_utils as T

    def chw(img):
        return torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0

    state = np.concatenate([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
    return {"observation.images.image": chw(obs["agentview_image"]),
            "observation.images.image2": chw(obs["robot0_eye_in_hand_image"]),
            "observation.state": torch.from_numpy(state)}


def run_episode(env, policy, pre, post, init_state, instruction, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(SETTLE_ACTION)
    policy.reset()
    for t in range(max_steps):
        batch = build_obs(obs)
        batch["task"] = instruction
        with torch.inference_mode():
            action = policy.select_action(pre(batch))
        action = post(action).squeeze(0).cpu().numpy().astype(np.float32)
        obs, _, _, _ = env.step(action)
        if env.check_success():
            return True, t + 1
    return False, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--episodes", type=int, default=50, help="per task (max 50)")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(NUM_TASKS)))
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--swap-instructions", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    policy, pre, post = load_policy(args.model_dir, args.device)
    results, t_start = [], time.time()
    for tid in args.tasks:
        own = task_info(tid)[0]
        given = task_info((tid + 1) % NUM_TASKS)[0] if args.swap_instructions else own
        env = make_env(tid)
        inits = eval_init_states(tid)
        n = min(args.episodes, len(inits))
        succ, steps = 0, []
        for ep in range(n):
            ok, k = run_episode(env, policy, pre, post, inits[ep], given, args.max_steps)
            succ += ok
            steps.append(k)
            print(f"  task {tid} ep {ep}: {'OK  ' if ok else 'FAIL'} ({k} steps)", flush=True)
        env.close()
        results.append(dict(task=tid, instruction=own, instruction_given=given, successes=succ,
                            episodes=n, success_rate=succ / n, mean_steps=float(np.mean(steps))))
        print(f"[task {tid}] {succ}/{n} = {succ / n:.0%}  {own}", flush=True)

    s = sum(r["successes"] for r in results)
    e = sum(r["episodes"] for r in results)
    print(f"\nOVERALL {s}/{e} = {s / max(1, e):.1%}  ({time.time() - t_start:.0f}s)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(label=args.label or str(args.model_dir), model_dir=str(args.model_dir),
                           swapped=args.swap_instructions, success_rate=s / max(1, e),
                           successes=s, episodes=e, per_task=results), f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

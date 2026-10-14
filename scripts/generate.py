"""Step 4: generate a LIBERO-Spatial dataset with MimicGen from a few source demos.

Each attempt: env.reset() samples a fresh scene layout from the task's BDDL
placement regions (never one of the 50 fixed eval init states), MimicGen
re-targets the source grasp segment to the new bowl pose and the place segment to
the new plate pose, executes it, and the episode is kept only if LIBERO's own
success check fires. Kept episodes are written in LIBERO's hdf5 layout so the
LeRobot converter from the previous project reads them unchanged.

One difference from the official LIBERO files, on purpose: obs[t] here is the
observation the action was chosen from (pre-step), which is what a policy sees at
eval time. LIBERO's create_dataset.py stores the post-step observation instead.

  python scripts/generate.py --task 0 --num 200
  -> data/generated/task0.hdf5   (+ task0_stats.json, task0_preview.mp4)
"""
import argparse
import json
import os
import time

import h5py
import imageio
import numpy as np

from libero_common import DATA_DIR, task_info
from mg_libero import LiberoMGEnv, MG_LiberoSpatial, libero_obs_record, make_task_spec

from mimicgen.datagen.data_generator import DataGenerator


def write_episode(grp, name, results, compress):
    obs = [libero_obs_record(o) for o in results["observations"]]
    ep = grp.create_group(name)
    og = ep.create_group("obs")
    for k in obs[0]:
        arr = np.stack([o[k] for o in obs])
        kw = dict(compression="gzip", compression_opts=4) if (compress and arr.dtype == np.uint8) else {}
        og.create_dataset(k, data=arr, **kw)
    actions = np.asarray(results["actions"])
    ep.create_dataset("actions", data=actions)
    ep.create_dataset("states", data=np.stack(results["states"]))
    T = len(actions)
    dones = np.zeros(T, np.uint8); dones[-1] = 1
    ep.create_dataset("dones", data=dones)
    ep.create_dataset("rewards", data=dones.copy())
    ep.attrs["num_samples"] = T
    ep.attrs["init_state"] = results["initial_state"]["states"]
    ep.attrs["src_demo_inds"] = np.array(results["src_demo_inds"])
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--num", type=int, default=200, help="successful episodes to keep")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--source", default=os.path.join(DATA_DIR, "source"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "generated"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-noise", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=400, help="discard longer episodes")
    ap.add_argument("--preview", type=int, default=3, help="episodes to render to mp4")
    ap.add_argument("--no-compress", action="store_true")
    args = ap.parse_args()
    max_attempts = args.max_attempts or args.num * 5

    np.random.seed(args.seed + 1000 * args.task)
    os.makedirs(args.out, exist_ok=True)
    lang, bddl, _ = task_info(args.task)
    src_path = os.path.join(args.source, f"task{args.task}.hdf5")
    with h5py.File(src_path, "r") as f:
        demo_keys = list(f["data"].keys())

    spec = make_task_spec(action_noise=args.action_noise)
    gen = DataGenerator(task_spec=spec, dataset_path=src_path, demo_keys=demo_keys)
    env = LiberoMGEnv(args.task)
    iface = MG_LiberoSpatial(env.base)

    out_path = os.path.join(args.out, f"task{args.task}.hdf5")
    prev_path = os.path.join(args.out, f"task{args.task}_preview.mp4")
    writer = imageio.get_writer(prev_path, fps=20) if args.preview else None
    kept, attempts, too_long, t0 = 0, 0, 0, time.time()
    lengths = []
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        grp.attrs["problem_info"] = json.dumps(dict(language_instruction=lang))
        grp.attrs["bddl_file_name"] = bddl
        grp.attrs["env_name"] = "Libero_Tabletop_Manipulation"
        grp.attrs["macros_image_convention"] = "opengl"
        grp.attrs["generator"] = "mimicgen"
        grp.attrs["source"] = src_path
        grp.attrs["source_demos"] = json.dumps(demo_keys)
        grp.attrs["task_spec"] = spec.serialize()
        while kept < args.num and attempts < max_attempts:
            attempts += 1
            res = gen.generate(env=env, env_interface=iface,
                               select_src_per_subtask=True,
                               transform_first_robot_pose=False,
                               interpolate_from_last_target_pose=True)
            T = len(res["actions"])
            if not res["success"]:
                continue
            if T > args.max_steps:
                too_long += 1
                continue
            write_episode(grp, f"demo_{kept}", res, compress=not args.no_compress)
            lengths.append(T)
            if writer is not None and kept < args.preview:
                for o in res["observations"][::2]:
                    # opengl convention: flip vertically for a human-readable video
                    writer.append_data(np.concatenate([o["agentview_image"][::-1],
                                                       o["robot0_eye_in_hand_image"][::-1]], axis=1))
            kept += 1
            f.flush()
            if kept % 10 == 0 or kept == args.num:
                el = time.time() - t0
                print(f"task {args.task}: kept {kept}/{args.num}  attempts {attempts}  "
                      f"rate {kept / attempts:.0%}  {el / attempts:.1f}s/attempt", flush=True)
        grp.attrs["num_demos"] = kept
        grp.attrs["total"] = int(sum(lengths))
    if writer is not None:
        writer.close()
    env.close()

    stats = dict(task=args.task, language=lang, kept=kept, attempts=attempts,
                 success_rate=kept / max(attempts, 1), too_long=too_long,
                 num_source_demos=len(demo_keys), mean_len=float(np.mean(lengths)) if lengths else None,
                 seconds=time.time() - t0)
    with open(os.path.join(args.out, f"task{args.task}_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

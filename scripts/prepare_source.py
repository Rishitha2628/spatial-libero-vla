"""Steps 1-2: pick source demos from LIBERO-Spatial and annotate them for MimicGen.

For each task, takes the first N demos of the official dataset, walks their
recorded MuJoCo states through the LIBERO env and stores, per timestep, the
"datagen_info" MimicGen parses: eef pose, bowl/plate poses, controller target pose
(from the recorded delta action), gripper action, and the grasp signal that
splits each demo into its two subtasks (grasp bowl | place on plate).

  python scripts/prepare_source.py --per-task 1     # 10 source demos total
  -> data/source/task{0..9}.hdf5
"""
import argparse
import json
import os

import h5py
import numpy as np

from libero_common import DATA_DIR, NUM_TASKS, task_info
from mg_libero import LiberoMGEnv, MG_LiberoSpatial


def annotate_demo(env, iface, states, actions):
    infos = []
    for s, a in zip(states, actions):
        env.base.sim.set_state_from_flattened(s)
        env.base.sim.forward()
        infos.append(iface.get_datagen_info(action=a))
    out = dict(
        eef_pose=np.stack([i.eef_pose for i in infos]),
        target_pose=np.stack([i.target_pose for i in infos]),
        gripper_action=np.stack([i.gripper_action for i in infos]),
        object_poses={k: np.stack([i.object_poses[k] for i in infos]) for k in infos[0].object_poses},
        subtask_term_signals={k: np.array([i.subtask_term_signals[k] for i in infos])
                              for k in infos[0].subtask_term_signals},
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=1, help="source demos per task")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(NUM_TASKS)))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "source"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for tid in args.tasks:
        lang, bddl, demo_path = task_info(tid)
        env = LiberoMGEnv(tid)
        iface = MG_LiberoSpatial(env.base)
        out_path = os.path.join(args.out, f"task{tid}.hdf5")
        with h5py.File(demo_path, "r") as src, h5py.File(out_path, "w") as dst:
            grp = dst.create_group("data")
            keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.per_task]
            for k in keys:
                states = src["data"][k]["states"][:]
                actions = src["data"][k]["actions"][:]
                assert states.shape[1] == env.base.sim.get_state().flatten().shape[0], "state size mismatch"
                info = annotate_demo(env, iface, states, actions)
                g = info["subtask_term_signals"]["grasp"]
                first_grasp = int(np.argmax(g)) if g.any() else -1
                print(f"task {tid} {k}: T={len(actions)} first grasp at {first_grasp}, "
                      f"grasped steps={int(g.sum())}")
                if first_grasp < 0:
                    print("  !! no grasp detected -- skipping")
                    continue
                ep = grp.create_group(k)
                ep.create_dataset("actions", data=actions)
                ep.create_dataset("states", data=states)
                ep.attrs["source_demo"] = f"{os.path.basename(demo_path)}:{k}"
                di = ep.create_group("datagen_info")
                for name in ("eef_pose", "target_pose", "gripper_action"):
                    di.create_dataset(name, data=info[name])
                for sub in ("object_poses", "subtask_term_signals"):
                    sg = di.create_group(sub)
                    for kk, v in info[sub].items():
                        sg.create_dataset(kk, data=v)
            grp.attrs["env_args"] = json.dumps(dict(task_id=tid, bddl=bddl, language=lang))
            grp.attrs["language"] = lang
            grp.attrs["total"] = sum(grp[k]["actions"].shape[0] for k in grp)
        env.close()
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Smoke test: native stack, EGL rendering, and a LIBERO-Spatial env.

  source env.sh && python scripts/check_env.py
"""
import os, sys
from libero_config_seed import seed_libero_config
seed_libero_config(os.path.join(os.environ.get("LIBERO_DIR", os.path.expanduser("~/LIBERO")),
                                "libero/libero"))
import mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

print(f"mujoco {mujoco.__version__}  MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

bm = benchmark.get_benchmark_dict()["libero_spatial"]()
names = bm.get_task_names()
print(f"libero_spatial: {len(names)} tasks")

t = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=True,
                 camera_heights=128, camera_widths=128, control_freq=20)
obs = env.reset()
assert obs["agentview_image"].shape == (128, 128, 3), obs["agentview_image"].shape
obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
env.close()
print("OK — native stack, EGL rendering and LIBERO-Spatial envs all working")

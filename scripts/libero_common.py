"""Shared LIBERO-Spatial helpers: config seeding, env construction, init states.

Import this before anything from `libero` -- LIBERO's first import calls input()
unless ~/.libero/config.yaml exists, and this module seeds it.
"""
import os

import numpy as np
import torch

from libero_config_seed import seed_libero_config

LIBERO_DIR = os.environ.get("LIBERO_DIR", os.path.expanduser("~/LIBERO"))
seed_libero_config(os.path.join(LIBERO_DIR, "libero/libero"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("SPATIAL_DATA_DIR", os.path.join(PROJ_DIR, "data"))
LIBERO_SPATIAL_DEMOS = os.path.join(DATA_DIR, "libero_spatial")

SUITE = "libero_spatial"
NUM_TASKS = 10
IMG_SIZE = 128
# Every LIBERO-Spatial task has the same goal: on(akita_black_bowl_1, plate_1).
BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
# Steps of zero action after reset so objects settle, as in LIBERO's eval protocol.
SETTLE_STEPS = 10
SETTLE_ACTION = np.array([0.0] * 6 + [-1.0])


def get_suite():
    return benchmark.get_benchmark_dict()[SUITE]()


def task_info(task_id):
    """(language instruction, bddl path, demo hdf5 path) for a task."""
    task = get_suite().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    demo = os.path.join(LIBERO_SPATIAL_DEMOS, task.name + "_demo.hdf5")
    return task.language, bddl, demo


def make_env(task_id, img_size=IMG_SIZE):
    _, bddl, _ = task_info(task_id)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size)


def eval_init_states(task_id):
    """The 50 fixed initial states LIBERO evaluates on.

    These ship with the LIBERO checkout (trusted), but were pickled with numpy
    arrays, which torch>=2.6's weights_only default refuses.
    """
    suite = get_suite()
    task = suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)

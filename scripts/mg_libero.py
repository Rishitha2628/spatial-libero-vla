"""MimicGen <-> LIBERO-Spatial adapter.

MimicGen generates against robomimic env wrappers and robosuite task classes;
LIBERO builds its scenes from BDDL files. Both sit on robosuite 1.4 with the same
OSC_POSE controller, so the bridge is small:

  LiberoMGEnv     -- the robomimic-style env API that DataGenerator / WaypointTrajectory
                     call (reset, get_state, get_observation, step, is_success).
  MG_LiberoSpatial-- the MimicGen env interface: eef pose, object poses, subtask signals.
                     Robot/controller conversions are inherited from RobosuiteInterface.

All ten LIBERO-Spatial tasks share one goal -- on(akita_black_bowl_1, plate_1) -- so
one interface and one two-subtask spec serve every task:
  1. grasp the bowl     (segment expressed relative to the bowl; ends at first grasp)
  2. place on the plate (segment expressed relative to the plate; ends at task success)
"""
import numpy as np

from libero_common import BOWL, PLATE, SETTLE_ACTION, SETTLE_STEPS, make_env

from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.env_interfaces.robosuite import RobosuiteInterface


class MG_LiberoSpatial(RobosuiteInterface):
    """`env` is the inner robosuite env (OffScreenRenderEnv.env)."""

    def get_object_poses(self):
        return dict(
            bowl=self.get_object_pose(obj_name=self.env.objects_dict[BOWL].root_body, obj_type="body"),
            plate=self.get_object_pose(obj_name=self.env.objects_dict[PLATE].root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        # Not robosuite's _check_grasp: the bowl is pinched by its rim, which often
        # touches only one fingerpad plus the finger bodies, so a pads-only test
        # misses real grasps. Here each side counts if its finger or its pad touches.
        g = self.env.robots[0].gripper.important_geoms
        bowl = self.env.objects_dict[BOWL]
        left = self.env.check_contact(g["left_finger"] + g["left_fingerpad"], bowl)
        right = self.env.check_contact(g["right_finger"] + g["right_fingerpad"], bowl)
        return dict(grasp=int(left and right))


class LiberoMGEnv:
    """robomimic EnvBase-style wrapper around a LIBERO OffScreenRenderEnv.

    Observations are cached from step()/reset() so get_observation() -- which
    WaypointTrajectory calls before every step -- doesn't re-render the cameras.
    """

    def __init__(self, task_id, img_size=128):
        self.task_id = task_id
        self.env = make_env(task_id, img_size)
        self.base = self.env.env
        self._obs = None

    def reset(self):
        self.env.reset()
        for _ in range(SETTLE_STEPS):
            self._obs, _, _, _ = self.env.step(SETTLE_ACTION)
        return self._obs

    def reset_to(self, flat_state):
        self._obs = self.env.set_init_state(flat_state)
        for _ in range(SETTLE_STEPS):
            self._obs, _, _, _ = self.env.step(SETTLE_ACTION)
        return self._obs

    def get_state(self):
        return dict(states=self.base.sim.get_state().flatten())

    def get_observation(self):
        return self._obs

    def step(self, action):
        self._obs, r, done, info = self.env.step(action)
        return self._obs, r, done, info

    def is_success(self):
        return dict(task=bool(self.env.check_success()))

    def close(self):
        self.env.close()


def make_task_spec(action_noise=0.05, grasp_offset=(5, 10), nn_k=3,
                   num_interpolation_steps=5, num_fixed_steps=0):
    """Two object-centric subtasks. Settings follow MimicGen's pick-and-place configs."""
    spec = MG_TaskSpec()
    common = dict(selection_strategy="nearest_neighbor_object",
                  selection_strategy_kwargs=dict(nn_k=nn_k),
                  action_noise=action_noise,
                  num_interpolation_steps=num_interpolation_steps,
                  num_fixed_steps=num_fixed_steps,
                  apply_noise_during_interpolation=False)
    spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp",
                     subtask_term_offset_range=grasp_offset, **common)
    spec.add_subtask(object_ref="plate", subtask_term_signal=None,
                     subtask_term_offset_range=None, **common)
    return spec


def libero_obs_record(obs):
    """Map a robosuite obs dict to LIBERO's hdf5 obs keys (see LIBERO scripts/create_dataset.py)."""
    import robosuite.utils.transform_utils as T
    ee_ori = T.quat2axisangle(obs["robot0_eef_quat"])
    return dict(
        agentview_rgb=obs["agentview_image"],
        eye_in_hand_rgb=obs["robot0_eye_in_hand_image"],
        ee_pos=obs["robot0_eef_pos"],
        ee_ori=ee_ori,
        ee_states=np.concatenate([obs["robot0_eef_pos"], ee_ori]),
        gripper_states=obs["robot0_gripper_qpos"],
        joint_states=obs["robot0_joint_pos"],
    )

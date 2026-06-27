import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp import reward, termination, observation
from active_adaptation.utils.motion import MotionDataset
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import (
    quat_apply_inverse,
    quat_apply,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    quat_from_angle_axis,
    yaw_quat,
)
from .base import Command
import re
import math
import gc
from typing import Sequence

from active_adaptation.envs.mdp.observations import random_noise
from active_adaptation.envs.mdp.commands.admittance import AdmittanceMassChain

def _match_indices(motion_names, asset_names, patterns, name_map=None, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            m = name_map.get(a, a) if name_map else a
            if m in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(m))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{m}' (idx {motion_names.index(m)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)

def _calc_exp_sigma(error : torch.Tensor, sigma_list : list[float], reduce_last_dim : bool = False):
    count = len(sigma_list)
    if reduce_last_dim:
        rewards = [torch.exp(- error / sigma).mean(dim=-1, keepdim=True) for sigma in sigma_list]
    else:
        rewards = [torch.exp(- error / sigma) for sigma in sigma_list]
    return sum(rewards) / count

def get_items_by_index(list, indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.tolist()
    return [list[i] for i in indexes]

def convert_dtype(dtype_str):
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': bool,
        'long': torch.long
    }
    if isinstance(dtype_str, str):
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return dtype_map[dtype_str]
    return dtype_str

class MotionTrackingCommand(Command):
    def __init__(self, env, dataset: dict,
                dataset_extra_keys: list[dict] = [],
                keypoint_map: dict = {
                    # "left_wrist_roll_link": "left_wrist_roll_rubber_hand",
                    # "right_wrist_roll_link": "right_wrist_roll_rubber_hand"
                },
                keypoint_patterns: list[str] = ["head_mimic", ".*_hand_mimic", ".*wrist_roll_link.*", ".*shoulder_yaw_link", ".*knee.*", ".*ankle_roll_link"],
                lower_keypoint_patterns: list[str] = [".*knee.*", ".*ankle_roll_link"],
                joint_patterns: list[str] = ["waist_*", ".*_hip_.*", ".*_knee.*",".*shoulder.*", ".*elbow.*", ".*wrist.*"],
                ignore_joint_patterns: list[str] = [".*ankle_roll_joint"],
                feet_patterns: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"],
                future_steps: Sequence[int] = (0, 2, 4, 8, 16),
                init_noise: dict[str, float] = {},
                reward_sigma: dict[str, list[float]] = {},
                student_train: bool = False,):
        super().__init__(env)

        if len(future_steps) < 2 or future_steps[0] != 0:
            raise ValueError("future_steps must start at 0 and contain at least one future frame")
        if any(curr <= prev for prev, curr in zip(future_steps, future_steps[1:])):
            raise ValueError("future_steps must be strictly increasing")
        self.future_steps = torch.tensor(future_steps, dtype=torch.int64, device=self.device)

        self.zero_init_prob = 1.0

        dataset_extra_keys = [
            {**k, 'dtype': convert_dtype(k['dtype'])} 
            for k in dataset_extra_keys
        ]

        self.student_train = student_train

        self.dataset = ProgressiveMultiMotionDataset(**dataset, env_size=self.num_envs, max_step_size=1000, dataset_extra_keys=dataset_extra_keys, device=self.device, ds_device=torch.device("cpu"), refresh_threshold=1000 * 20)
        self.dataset.set_limit(self.asset.data.soft_joint_pos_limits, self.asset.data.soft_joint_vel_limits, self.asset.joint_names)

        # bodies for full‑body keypoint tracking
        self.keypoint_patterns = keypoint_patterns
        self.lower_keypoint_patterns = lower_keypoint_patterns
        self.keypoint_map = keypoint_map
        self.keypoint_idx_motion, self.keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.lower_keypoint_idx_motion, self.lower_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.lower_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        # joints for full‑body joint tracking
        self.joint_patterns = joint_patterns
        self.joint_idx_motion, self.joint_idx_asset = _match_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            self.joint_patterns,
            device=self.device
        )
        
        self.feet_patterns = feet_patterns
        self.feet_idx_motion, self.feet_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.feet_patterns,
            device=self.device
        )

        # all joints except ankles
        self.ignore_joint_patterns = ignore_joint_patterns
        all_j_m, all_j_a = [], []
        for j in self.asset.joint_names:
            if j in self.dataset.joint_names and not any(re.match(p, j) for p in self.ignore_joint_patterns):
                all_j_m.append(self.dataset.joint_names.index(j))
                all_j_a.append(self.asset.joint_names.index(j))
        self.all_joint_idx_dataset, self.all_joint_idx_asset = all_j_m, all_j_a
        self.all_joint_idx_dataset = torch.tensor(self.all_joint_idx_dataset, device=self.device)
        self.all_joint_idx_asset = torch.tensor(self.all_joint_idx_asset, device=self.device)

        self.last_reset_env_ids = None

        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._cum_root_pos_scale = 0.3
        self._cum_keypoint_scale = 0.25
        self._cum_orientation_scale = 0.7
        self.feet_standing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        self.lengths = torch.full((self.num_envs,), 1, dtype=torch.int32, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boot_indicator = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.boot_indicator_max = 25

        self.joint_pos_last = torch.zeros(self.num_envs, len(self.joint_idx_asset), device=self.device)
        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()

        ## init noise
        self.init_noise_params = init_noise
        ## reward sigma
        self.reward_sigma = reward_sigma

    def sample_init(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        lengths = self.lengths[env_ids]
        self.last_reset_env_ids = env_ids
        # resample motion
        lengths = self.dataset.reset(env_ids)
        
        max_start = lengths - self.future_steps[-1] - 1
        rand_float = torch.rand(env_ids.shape[0], device=env_ids.device) * 0.75  # [0,0.75)
        offsets = (rand_float * max_start.to(torch.float32)).floor().int()
        t[:] = offsets * (torch.rand(env_ids.shape[0], device=env_ids.device) > self.zero_init_prob)

        self.lengths[env_ids] = lengths
        self.t[env_ids] = t

        motion = self.dataset.get_slice(env_ids, self.t[env_ids], 1)

        # set robot state
        self.sample_init_robot(env_ids, motion)
        return None

    def sample_init_robot(self, env_ids: Sequence[int], motion, lift_height: float = 0.04):
        # Get subsets for the current envs
        init_root_state = self.init_root_state[env_ids].clone()
        init_joint_pos = self.init_joint_pos[env_ids].clone()
        init_joint_vel = self.init_joint_vel[env_ids].clone()
        env_origins = self.env.scene.env_origins[env_ids]
        num_envs = len(env_ids)

        # Extract motion data
        motion_root_pos = motion.root_pos_w[:, 0]
        motion_root_quat = motion.root_quat_w[:, 0]
        motion_root_lin_vel = motion.root_lin_vel_w[:, 0]
        motion_root_ang_vel = motion.root_ang_vel_w[:, 0]
        motion_joint_pos = motion.joint_pos[:, 0]
        motion_joint_vel = motion.joint_vel[:, 0]

        # -------- root state ----------------------------------------------------
        init_root_state[:, :3] = env_origins + motion_root_pos
        init_root_state[:, 2] += lift_height
        root_pos_noise = torch.randn_like(init_root_state[:, :3]).clamp(-1, 1) * self.init_noise_params["root_pos"]
        root_pos_noise[:, 2].clamp_min_(0.0)
        init_root_state[:, :3] += root_pos_noise

        init_root_state[:, 3:7] = motion_root_quat
        random_axis = torch.rand(num_envs, 3, device=self.device)
        random_angle = torch.randn(num_envs, device=self.device).clamp(-1, 1) * self.init_noise_params["root_ori"]
        random_quat = quat_from_angle_axis(random_angle, random_axis)
        init_root_state[:, 3:7] = quat_mul(random_quat, init_root_state[:, 3:7])

        init_root_state[:, 7:10] = motion_root_lin_vel
        lin_vel_noise = torch.randn_like(init_root_state[:, 7:10]).clamp(-1, 1) * self.init_noise_params["root_lin_vel"]
        init_root_state[:, 7:10] += lin_vel_noise
        
        init_root_state[:, 10:13] = motion_root_ang_vel
        ang_vel_noise = torch.randn_like(init_root_state[:, 10:13]).clamp(-1, 1) * self.init_noise_params["root_ang_vel"]
        init_root_state[:, 10:13] += ang_vel_noise

        # -------- joint state ----------------------------------------------------
        init_joint_pos[:, self.all_joint_idx_asset] = motion_joint_pos[:, self.all_joint_idx_dataset]
        init_joint_vel[:, self.all_joint_idx_asset] = motion_joint_vel[:, self.all_joint_idx_dataset]
        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_noise_params["joint_pos"]
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_noise_params["joint_vel"]
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # Apply the calculated states to the simulation
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)

        self.joint_pos_last[env_ids] = init_joint_pos[:, self.joint_idx_asset]
        self.joint_pos_boot_protect[env_ids] = init_joint_pos

        self.asset.write_joint_position_to_sim(init_joint_pos, env_ids=env_ids)
        self.asset.set_joint_position_target(init_joint_pos, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(init_joint_vel, env_ids=env_ids)

        self.asset.write_data_to_sim()
    
    def reset(self, env_ids):
        self.finished[env_ids] = False
        self.boot_indicator[env_ids] = self.boot_indicator_max
        self._cum_error[env_ids] = 0.0

    @observation
    def target_pos_b_obs(self):
        current_pos = self.asset.data.root_pos_w.unsqueeze(1) - self.env.scene.env_origins.unsqueeze(1)
        current_quat = self.asset.data.root_quat_w.unsqueeze(1)
        target_pos_b = quat_apply_inverse(
            current_quat,
            (self._motion.root_pos_w - current_pos)
        )
        return target_pos_b.reshape(self.num_envs, -1)
    def target_pos_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))
    
    @observation
    def target_linvel_b_obs(self):
        target_linvel_b = quat_apply_inverse(self.asset.data.root_quat_w.unsqueeze(1), self._motion.root_lin_vel_w)
        return target_linvel_b.reshape(self.num_envs, -1)
    def target_linvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_projected_gravity_b(self):
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        g_b = quat_apply_inverse(self._motion.root_quat_w, gravity)  # [N, S, 3]
        return g_b.reshape(self.num_envs, -1)

    def target_projected_gravity_b_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_keypoints_b_obs(self):
        target_keypoints_b = self._motion.body_pos_b[:, :, self.keypoint_idx_motion]
        return target_keypoints_b.reshape(self.num_envs, -1)
    def target_keypoints_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))
    
    @observation
    def target_keypoints_diff_b_obs(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset] - self.env.scene.env_origins.unsqueeze(1)
        target_w = self._motion.body_pos_w[:, :, self.keypoint_idx_motion]
        diff_w = target_w - actual_w.unsqueeze(1)
        diff_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1).unsqueeze(1),
            diff_w
        )
        return diff_b.reshape(self.num_envs, -1)
    def target_keypoints_diff_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1]).repeat(len(self.future_steps))

    @observation
    def relative_quat_obs(self):
        relative_quat = axis_angle_from_quat(quat_mul(
            self._motion.root_quat_w,
            quat_conjugate(self.asset.data.root_quat_w.unsqueeze(1).expand_as(self._motion.root_quat_w))
        ))
        return relative_quat.reshape(self.num_envs, -1)
    def relative_quat_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[-1, 1, -1]
        ).repeat(len(self.future_steps))

    @observation
    def target_joint_pos_obs(self):
        return self._motion.joint_pos.reshape(self.num_envs, -1)
    def target_joint_pos_obs_sym(self):
        return sym_utils.joint_space_symmetry(self.asset, self.dataset.joint_names).repeat(len(self.future_steps))


    @observation
    def current_keypoint_b(self):
        actual_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        actual_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        return actual_b.reshape(self.num_envs, -1)
    def current_keypoint_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation
    def current_keypoint_vel_b(self):
        actual_vel_w = self.asset.data.body_lin_vel_w[:, self.keypoint_idx_asset]
        actual_vel_b = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            actual_vel_w
        )
        return actual_vel_b.reshape(self.num_envs, -1)
    def current_keypoint_vel_b_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])
    
    @observation
    def boot_indicator_state(self):
        return self.boot_indicator / self.boot_indicator_max
    def boot_indicator_state_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1.])

    @reward
    def root_pos_tracking(self):
        current_pos = self.asset.data.root_pos_w
        target_pos = self.reward_root_pos_w
        diff = target_pos - current_pos
        error = diff.norm(dim=-1, keepdim=True)
        self._cum_error[:, 0:1] = error / self._cum_root_pos_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_pos"])

    @reward
    def root_vel_tracking(self):
        current_linvel_w = self.asset.data.root_lin_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_linvel_w = self._motion.root_lin_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_linvel_b = quat_apply_inverse(current_quat, current_linvel_w)
        ref_linvel_b = quat_apply_inverse(ref_quat, ref_linvel_w)
        diff = ref_linvel_b - current_linvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_vel"])

    @reward
    def root_rot_tracking(self):
        current_quat = self.asset.data.root_quat_w
        target_quat = self.reward_root_quat_w
        diff = axis_angle_from_quat(quat_mul(
            target_quat,
            quat_conjugate(current_quat)
        ))
        error = torch.norm(diff, dim=-1, keepdim=True)
        self._cum_error[:, 1:2] = error / self._cum_orientation_scale
        return _calc_exp_sigma(error, self.reward_sigma["root_rot"])
    
    @reward
    def root_ang_vel_tracking(self):
        current_angvel_w = self.asset.data.root_ang_vel_w
        current_quat = self.asset.data.root_quat_w
        ref_angvel_w = self._motion.root_ang_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_angvel_b = quat_apply_inverse(current_quat, current_angvel_w)
        ref_angvel_b = quat_apply_inverse(ref_quat, ref_angvel_w)
        diff = ref_angvel_b - current_angvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["root_ang_vel"])

    @reward
    def keypoint_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, self.reward_sigma["keypoint"])
    
    @reward
    def lower_keypoint_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.lower_keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.lower_keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["lower_keypoint"])

    @reward
    def joint_pos_tracking(self):
        actual = self.asset.data.joint_pos[:, self.joint_idx_asset]
        target = self._motion.joint_pos[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_pos"])

    @reward
    def joint_vel_tracking(self):
        current_joint_pos = self.asset.data.joint_pos[:, self.joint_idx_asset]
        vel_diff = (current_joint_pos - self.joint_pos_last) / self.env.step_dt
        self.joint_pos_last[:] = current_joint_pos

        target = self._motion.joint_vel[:, 0, self.joint_idx_motion]
        error = (target - vel_diff).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["joint_vel"])

    def update_reward_target(self):
        delta_yaw = quat_mul(yaw_quat(self.asset.data.root_quat_w), quat_conjugate(yaw_quat(self._motion.root_quat_w[:, 0])))
        tgt_rel = self._motion.body_pos_w[:, 0] - self._motion.root_pos_w[:, 0].unsqueeze(1)
        self.reward_keypoints_w = quat_apply(delta_yaw.unsqueeze(1), tgt_rel) + self.asset.data.root_pos_w.unsqueeze(1)
        self.reward_keypoints_w[:, 2] = self._motion.body_pos_w[:, 0, 2]

        if not self.student_train:
            self.reward_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
            self.reward_root_quat_w = self._motion.root_quat_w[:, 0]
        else:
            steps = 50 # calc t+50 target root pos/rot from current root pos/rot
            # prepare future root pos/rot cache
            if hasattr(self, 'ts_root_pos_w') is False:
                self.ts_root_pos_w = torch.zeros(self.num_envs, steps, 3, device=self.device, dtype=torch.float32)
                self.ts_root_quat_w = torch.zeros(self.num_envs, steps, 4, device=self.device, dtype=torch.float32)
            # update only for reset envs
            if self.last_reset_env_ids is not None:
                future_motion = self.dataset.get_slice(self.last_reset_env_ids, self.t[self.last_reset_env_ids], steps=steps)
                self.ts_root_pos_w[self.last_reset_env_ids] = future_motion.root_pos_w + self.env.scene.env_origins[self.last_reset_env_ids].unsqueeze(1)
                self.ts_root_quat_w[self.last_reset_env_ids] = future_motion.root_quat_w
            # get current root pos/rot from cache
            reward_pos = self.ts_root_pos_w[:, 0].clone()
            reward_quat = self.ts_root_quat_w[:, 0].clone()
            # roll forward the cache
            self.ts_root_pos_w[:, :-1] = self.ts_root_pos_w[:, 1:]
            self.ts_root_quat_w[:, :-1] = self.ts_root_quat_w[:, 1:]
            # compute target root pos/rot at t+steps
            current_pos_t = self.asset.data.root_pos_w
            current_quat_t = self.asset.data.root_quat_w
            ref_motion_plus = self.dataset.get_slice(None, self.t, steps=torch.tensor([steps], device=self.device, dtype=torch.int64))
            ref_pos_t = self._motion.root_pos_w[:, 0]
            ref_pos_t_plus = ref_motion_plus.root_pos_w[:, 0]
            ref_quat_t = self._motion.root_quat_w[:, 0]
            ref_quat_t_plus = ref_motion_plus.root_quat_w[:, 0]

            delta_yaw = quat_mul(yaw_quat(current_quat_t), quat_conjugate(yaw_quat(ref_quat_t)))
            self.ts_root_pos_w[:, -1] = quat_apply(delta_yaw, (ref_pos_t_plus - ref_pos_t)) + current_pos_t
            self.ts_root_pos_w[:, -1, 2] = ref_pos_t_plus[:, 2]  # recover z axis
            self.ts_root_quat_w[:, -1] = quat_mul(delta_yaw, ref_quat_t_plus)

            self.reward_root_pos_w = reward_pos
            self.reward_root_quat_w = reward_quat

    def before_update(self):
        self.t = torch.clamp_max(self.t + 1, self.lengths - 1)
        self.finished[:] = self.t >= self.lengths - 1
        self.boot_indicator[:] = torch.clamp_min(self.boot_indicator - 1, 0)

        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)
        self.update_reward_target()

        feet_vel = (self._motion.body_pos_w[:, 1, self.feet_idx_motion] - self._motion.body_pos_w[:, 0, self.feet_idx_motion]) / ((self.future_steps[1] - self.future_steps[0]) * self.env.step_dt) # [N, 2, 3]
        self.feet_standing = (feet_vel[:, :, :2].norm(dim=-1, keepdim=False) < 0.1)

    def update(self):
        self.dataset.update()
        if self.last_reset_env_ids is not None:
            self.last_reset_env_ids = None

    def debug_draw(self):
        root_pos = self.asset.data.root_pos_w    # [N,1,3]
        root_quat = self.asset.data.root_quat_w  # [N,1,4]
        target_root_quat = self._motion.root_quat_w[:, 0, :]
        heading_rel = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        heading_world = quat_apply(root_quat, heading_rel)
        heading_world_target = quat_apply(target_root_quat, heading_rel)

        # —— original world‐frame drawing —— 
        target_keypoints_w = self.reward_keypoints_w[:, self.keypoint_idx_motion]
        robot_keypoints_w = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]

        # draw points and error vectors
        self.env.debug_draw.point(
            target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1)
        )
        self.env.debug_draw.point(
            robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1)
        )
        self.env.debug_draw.vector(
            robot_keypoints_w.reshape(-1, 3),
            (target_keypoints_w - robot_keypoints_w).reshape(-1, 3),
            color=(0, 0, 1, 1)
        )
        
        self.env.debug_draw.vector(
            root_pos.reshape(-1, 3),
            heading_world.reshape(-1, 3),
            color=(0, 0, 1, 2)
        )
        
        self.env.debug_draw.vector(
            root_pos.reshape(-1, 3),
            heading_world_target.reshape(-1, 3),
            color=(1, 0, 0, 2)
        )

from active_adaptation.utils.math import normalize
from .utils import TemporalLerp, clamp_norm, create_mapping, rand_points_disk, rand_points_isotropic, random_uniform
import os

class MotionTrackingCommand_impedance(MotionTrackingCommand):
    def __init__(
        self,
        env,
        *args,
        max_force: float = 30.0,
        compliance: bool = True,
        net_force_limit: float = 30.0,
        net_torque_limit: float = 20.0,
        **kwargs,
    ):
        super().__init__(env, *args, **kwargs)

        force_apply_pattern = [".*shoulder_yaw_link", ".*wrist_roll_link", ".*hand_mimic"]
        self.force_apply_idx_motion, self.force_apply_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            force_apply_pattern,
            name_map=self.keypoint_map,
        )
        self.force_in_keypoint_idx = create_mapping(self.keypoint_idx_asset, self.force_apply_idx_asset, self.device)
        self.num_force_bodies = self.force_apply_idx_asset.shape[0]

        self.max_force = max_force
        self.net_force_limit  = torch.as_tensor(net_force_limit,  dtype=torch.float32, device=self.device)
        self.net_torque_limit = torch.as_tensor(net_torque_limit, dtype=torch.float32, device=self.device)
        self.compliance = compliance

        # force origin samples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "../../../..")
        hand_samples_path = os.path.join(project_root, "scripts/data_process/hand_grid_samples.pt")
        self.force_origin_samples_raw = torch.load(hand_samples_path, map_location=self.device)
        self.force_origin_samples_left = self.force_origin_samples_raw["left"].to(self.device)
        self.force_origin_samples_right = self.force_origin_samples_raw["right"].to(self.device)

        self.force_origin_sample_prob = 0.5
        self.force_origin_samples = self.force_origin_samples_left.shape[0]

        self.kp_range = (5.0, 250.0)
        self.kp_slope_range = (-5.0, 5.0)
        self.zero_kp_slope_prob = 0.5
        self.kp_time_range = (25, 100)
        self.force_time_range = (20, 200)
        self.ramping_time_range = (25, 100)
        self.force_origin_transit_time_range = (25, 100)

        self.skip_ref = False

        with torch.device(self.device):
            self.force_type_probs = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15]) # [zero_force, full_force, left_full, right_full, partial_force]
            self.left_mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)[None, :, None]
            self.right_mask = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)[None, :, None]
            self.force_partial_single_prob = 0.5

            # force threshold sample
            self.force_safe_bounds = (5.0, 15.0)
            self.force_penalty_offset = 10.0
            self.force_safe_default = 10.0
            self.force_safe_limit_tl = TemporalLerp(
                shape=(self.num_envs, 1),
                device=self.device,
                easing="linear",
                clamp=self.force_safe_bounds
            )
            self.force_safe_limit_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32) # next time to change safe limit

            # sample control
            self.force_type = torch.zeros(self.num_envs, dtype=torch.int32)
            self.force_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32)
            self.force_enable = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.bool) # enable flag

            # force spring / ramping
            self.force_kp_scaled = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.float32) # scaled spring stiffness (multiplied by alpha)
            self.force_kp_sample_timer = torch.zeros(self.num_envs, dtype=torch.int32) # next time to change slope
            self.force_kp_ramping_down = torch.zeros(self.num_envs, dtype=torch.bool) # is ramping down
            self.force_kp_tl = TemporalLerp(
                shape=(self.num_envs, self.num_force_bodies, 1),
                device=self.device,
                easing="linear",
                clamp=self.kp_range
            )

            # admittance tracking
            self.ref_pos_b_prev = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_keypoint_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # target keypoints in world frame
            self.force_keypoint_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # target keypoints in body frame
            self.force_keypoint_w_prev = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_keypoint_vel_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)

            # spring origin b -> origin w -> dir w
            self.force_origin_tl = TemporalLerp(
                shape=(self.num_envs, self.num_force_bodies, 3),
                device=self.device,
                easing="linear"
            )
            self.force_origin_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # origin position of spring in world frame
            self.force_dir_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            # spring force
            self.force_applied_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # applied force
            self.force_applied_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # applied force in body frame
            self.force_expected_w = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # expected apply force
            self.force_expected_b = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32) # expected apply force in body frame

            # force apply
            self.force_pos_delta = torch.zeros(self.num_envs, self.num_force_bodies, 3, dtype=torch.float32)
            self.force_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, dtype=torch.float32)
            self.torque_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, dtype=torch.float32)

            # student obs
            student_body_pattern = ["torso"]
            self.student_body_idx_motion, self.student_body_idx_asset = _match_indices(
                self.dataset.body_names,
                self.asset.body_names,
                student_body_pattern,
                debug=True
            )
            self.torso_idx_motion = self.student_body_idx_motion[0].item()
            self.torso_idx_asset = self.student_body_idx_asset[0].item()

            # baseline: random perturbation forces (non-compliant mode)
            self.perturb_force = TemporalLerp(
                shape=(self.num_envs, 2, 3),
                device=self.device,
                easing="linear",
            )

        self.force_alpha = 1.0

        self.configure_admittance()

    def configure_admittance(self):
        self.admit = AdmittanceMassChain(
            num_envs=self.num_envs,
            num_points=self.num_force_bodies,
            dt=self.env.physics_dt, # !important
            mixed_loop_steps=1,
            device=self.device,
            mass=0.1,
            damping=2.0,
            vel_clip=4.0,
            acc_clip=1000.0
        )

    def sample_init(self, env_ids: torch.Tensor):
        super().sample_init(env_ids)
        self.force_reset(env_ids)
        return None

    def force_reset(self, env_ids: torch.Tensor, refresh_time: bool = True):
        self.force_type[env_ids] = 0
        self.force_applied_w[env_ids] = 0
        self.force_applied_b[env_ids] = 0
        self.force_enable[env_ids] = 0
        self.force_origin_tl.reset(env_ids)
        self.force_kp_tl.reset(env_ids)
        self.force_kp_ramping_down[env_ids] = False
        self.perturb_force.reset(env_ids, value=0.0)
        
        if refresh_time:
            self.force_sample_timer[env_ids] = random_uniform(env_ids.shape[0], 10, 60, self.device).int()
            self.force_safe_limit_tl.reset(env_ids)
            self.force_safe_limit_tl.set(env_ids, start=self.force_safe_default, end=self.force_safe_default, total_steps=1)
            self.force_safe_limit_sample_timer[env_ids] = 0
    
    def force_schedule(self):
        # procedure:
        # 0. update force kp
        #    - if force_kp_sample_timer is done, resample force_kp_slope and force_kp_active
        # 1. check if force is ramping down, if so, update the force
        #    - if ramping down is done, reset force for these envs
        #    - if not done, update the force_kp_active and force_ramping_down
        # 2. check if (time is done), if so, start ramping down
        # 3. check if (time is done) & (ramping down is done), if so, resample force

        self.force_kp_sample_timer -= 1
        self.force_sample_timer -= 1
        self.force_safe_limit_sample_timer -= 1

        # -0 update force safe limit
        need_resample_safe_limit = self.force_safe_limit_sample_timer < 0
        if need_resample_safe_limit.any():
            resample_envs = need_resample_safe_limit.nonzero(as_tuple=False).squeeze(-1)
            self.force_safe_limit_sample_timer[resample_envs] = random_uniform(resample_envs.shape[0], 100, 200, self.device).int()
            safe_limit = random_uniform((resample_envs.shape[0], 1), self.force_safe_bounds[0], self.force_safe_bounds[1], self.device)
            delta_time = random_uniform(resample_envs.shape[0], 25, 100, self.device).int()
            self.force_safe_limit_tl.set(resample_envs, end=safe_limit, total_steps=delta_time)

        # -0 update force kp
        need_resample_kp = (self.force_kp_sample_timer < 0) & (~self.force_kp_ramping_down)
        if need_resample_kp.any():
            resample_envs = need_resample_kp.nonzero(as_tuple=False).squeeze(-1)
            kp_next_time = random_uniform(resample_envs.shape[0], self.kp_time_range[0], self.kp_time_range[1], self.device).int()
            self.force_kp_sample_timer[resample_envs] = kp_next_time
            kp_zero_slope = torch.rand(resample_envs.shape[0], 2, 1, 1, device=self.device) < self.zero_kp_slope_prob
            kp_slope = random_uniform((resample_envs.shape[0], 2, 1, 1), self.kp_slope_range[0], self.kp_slope_range[1], self.device) * (~kp_zero_slope)
            kp_delta = (kp_slope[:, 0] * self.left_mask + kp_slope[:, 1] * self.right_mask) * kp_next_time[:, None, None]  # [N, 1, 1]
            self.force_kp_tl.set(resample_envs, delta=kp_delta, total_steps=kp_next_time)

        # -1
        # deal with force ramping down
        finished_ramping_down = self.force_kp_ramping_down & self.force_kp_tl.mask_done
        if finished_ramping_down.any():
            finished_ramping_down_envs = finished_ramping_down.nonzero(as_tuple=False).squeeze(-1)
            self.force_kp_ramping_down[finished_ramping_down_envs] = False
            self.force_reset(finished_ramping_down_envs, refresh_time=False)  # reset force for these envs

        # -2
        # time done
        time_done = self.force_sample_timer < 0
        # force required
        force_required_mask = self.force_enable.any(dim=(1,2))
        
        # ramping force when (1) time is done & (2) not ramping now & (3) currently have force
        should_start_ramping_down = time_done & (~self.force_kp_ramping_down) & force_required_mask
        if should_start_ramping_down.any():
            ramping_down_envs = should_start_ramping_down.nonzero(as_tuple=False).squeeze(-1)
            self.force_kp_ramping_down[ramping_down_envs] = True
            steps = random_uniform(ramping_down_envs.shape[0], self.ramping_time_range[0], self.ramping_time_range[1], self.device).int()
            self.force_kp_tl.set(env_ids=ramping_down_envs, end=0.0, total_steps=steps)

        # -3 deal with new force sampling
        need_resample = time_done & (~self.force_kp_ramping_down) # recompute current ramping envs
        if need_resample.any():
            need_resample_envs = need_resample.nonzero(as_tuple=False).squeeze(-1)
            force_type = torch.multinomial(self.force_type_probs, need_resample_envs.shape[0], replacement=True)
            zero_force = force_type == 0
            full_force = force_type == 1
            left_full = force_type == 2
            right_full = force_type == 3
            partial_force = force_type == 4
            
            force_enable = torch.zeros(need_resample_envs.shape[0], self.num_force_bodies, 1, dtype=torch.bool, device=self.device)
            force_enable[zero_force, :] = 0
            force_enable[full_force, :] = 1
            force_enable[left_full, :] = self.left_mask
            force_enable[right_full, :] = self.right_mask
            force_enable[partial_force, :] = torch.rand_like(force_enable[partial_force, :], dtype=torch.float32) <= self.force_partial_single_prob
            self.force_enable[need_resample_envs] = force_enable

            kp_left = random_uniform((need_resample_envs.shape[0], 1, 1), self.kp_range[0], self.kp_range[1], self.device)
            kp_right = random_uniform((need_resample_envs.shape[0], 1, 1), self.kp_range[0], self.kp_range[1], self.device)
            kp = (kp_left * self.left_mask + kp_right * self.right_mask) * force_enable
            self.force_kp_tl.set(need_resample_envs, end=kp, total_steps=1)

            self.force_sample_timer[need_resample_envs] = random_uniform(need_resample_envs.shape[0], self.force_time_range[0], self.force_time_range[1], self.device).int()
            self.force_reset_origin(need_resample_envs)

            # deal with force delta pos
            self.force_pos_delta[need_resample_envs] = rand_points_isotropic(need_resample_envs.shape[0], self.num_force_bodies, r_max=0.05, device=self.device)

        self.force_kp_scaled = self.force_kp_tl.current * self.force_enable

    def force_reset_origin(self, env_ids: torch.Tensor):
        pos_w = self.asset.data.body_pos_w[env_ids][:, self.force_apply_idx_asset, :]
        root_pos = self.asset.data.root_pos_w[env_ids, :]
        root_quat = self.asset.data.root_quat_w[env_ids, :]
        pos_b_start = quat_apply_inverse(root_quat.unsqueeze(1), pos_w - root_pos.unsqueeze(1))
        pos_b_end_left = pos_b_start[:, self.left_mask.reshape(-1), :]
        pos_b_end_right = pos_b_start[:, self.right_mask.reshape(-1), :]
        # in some env, we will sample a pulling force but not on partial force
        need_left_sample = ((torch.rand(env_ids.shape[0], device=self.device) < self.force_origin_sample_prob) & (self.force_type[env_ids] != 4)).nonzero(as_tuple=False).squeeze(-1)
        need_right_sample = ((torch.rand(env_ids.shape[0], device=self.device) < self.force_origin_sample_prob) & (self.force_type[env_ids] != 4)).nonzero(as_tuple=False).squeeze(-1)

        def sample_origin(source: torch.Tensor, env_ids: torch.Tensor):
            idx = (torch.rand(env_ids.shape[0], device=self.device) * self.force_origin_samples).floor().int()
            link_pos_in_torso = source[idx]
            torso_pos = self.asset.data.body_pos_w[env_ids, self.torso_idx_asset].unsqueeze(1)
            torso_quat = self.asset.data.body_quat_w[env_ids, self.torso_idx_asset].unsqueeze(1)

            root_pos = self.asset.data.root_pos_w[env_ids, :].unsqueeze(1)
            root_quat = self.asset.data.root_quat_w[env_ids, :].unsqueeze(1)

            link_pos_w = quat_apply(torso_quat, link_pos_in_torso) + torso_pos
            return quat_apply_inverse(root_quat, link_pos_w - root_pos)

        pos_b_end_left[need_left_sample] = sample_origin(self.force_origin_samples_left, env_ids[need_left_sample])
        pos_b_end_right[need_right_sample] = sample_origin(self.force_origin_samples_right, env_ids[need_right_sample])

        pos_b_end = torch.zeros_like(pos_b_start)
        pos_b_end[:, self.left_mask.reshape(-1), :] = pos_b_end_left
        pos_b_end[:, self.right_mask.reshape(-1), :] = pos_b_end_right

        transit_steps = random_uniform(env_ids.shape[0], self.force_origin_transit_time_range[0], self.force_origin_transit_time_range[1], self.device).int()

        # pos_b_end += rand_points_isotropic(env_ids.shape[0], self.num_force_bodies, r_max=0.05, device=self.device)

        self.force_origin_tl.set(env_ids=env_ids, start=pos_b_start, end=pos_b_end, total_steps=transit_steps)

    def force_update_origin_and_target(self):
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w
        root_quat_exp = root_quat.unsqueeze(1)

        # spring origin is stored in the root frame
        force_origin_b = self.force_origin_tl.current
        self.force_origin_w[:] = quat_apply(root_quat_exp, force_origin_b) + root_pos_w

        # reference keypoints expressed directly in the robot root frame
        ref_point_b = quat_apply_inverse(root_quat_exp, self.reward_keypoints_w[:, self.force_apply_idx_motion] - root_pos_w)

        # calc ref point vel
        if self.last_reset_env_ids is not None:
            self.ref_pos_b_prev[self.last_reset_env_ids] = ref_point_b[self.last_reset_env_ids]
        ref_point_vel_b = (ref_point_b - self.ref_pos_b_prev) / self.env.step_dt
        self.ref_pos_b_prev[:] = ref_point_b

        force_dir_b = normalize(ref_point_b - force_origin_b)

        self.force_dir_w[:] = quat_apply(root_quat_exp, force_dir_b)

        current_point_w = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        current_point_b = quat_apply_inverse(root_quat_exp, current_point_w - root_pos_w)

        current_point_vel_w = self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset]
        root_lin_vel_w = self.asset.data.root_lin_vel_w.unsqueeze(1)
        rel_vel_w = current_point_vel_w - root_lin_vel_w

        root_ang_vel_w = self.asset.data.root_ang_vel_w
        omega_b = quat_apply_inverse(root_quat, root_ang_vel_w).unsqueeze(1)
        omega_b = omega_b.expand(-1, current_point_b.shape[1], -1)
        rel_vel_b = quat_apply_inverse(root_quat_exp, rel_vel_w)

        # deal with init state
        if self.last_reset_env_ids is not None:
            self.admit.reset(
                self.last_reset_env_ids,
                x0_b=ref_point_b[self.last_reset_env_ids],
                v0_b=ref_point_vel_b[self.last_reset_env_ids],
            )

        # driving force in root frame
        force_limit = self.force_safe_limit_tl.current.unsqueeze(1)         # [N, 1, 1]
        K_p_drive = (force_limit / 0.05)                                    # [N, 1, 1] 5cm
        K_d_drive = 2.0 * torch.sqrt(K_p_drive * 0.1)                       # critical damping assuming mass=0.1kg

        ref_point_b_exp = ref_point_b.unsqueeze(0)
        ref_point_vel_b_exp = ref_point_vel_b.unsqueeze(0)
        force_origin_b_exp = force_origin_b.unsqueeze(0)
        force_dir_b_exp = force_dir_b.unsqueeze(0)

        # use 4 substep to integrate in root frame
        for _ in range(4):
            admit_point_b = self.admit.x  # current integrate pos (root frame)
            admit_point_vel_b = self.admit.v  # current integrate vel (root frame)
            # driving force
            F_drive_b = clamp_norm(
                K_p_drive * (ref_point_b_exp - admit_point_b) +
                K_d_drive * (ref_point_vel_b_exp - admit_point_vel_b),
                max=force_limit
            )
            # external force
            F_ext_b = clamp_norm(
                self.force_kp_scaled * self.project_pos_diff(
                    force_origin_b_exp - admit_point_b,
                    force_dir=force_dir_b_exp,
                ),
                self.max_force * self.force_alpha
            )

            self.admit.step(F_drive_b=F_drive_b, F_ext_b=F_ext_b)

        force_keypoint_b = self.admit.x[0]

        self.force_keypoint_b[:] = force_keypoint_b
        self.force_keypoint_w[:] = quat_apply(root_quat_exp, force_keypoint_b) + root_pos_w

        # get other force target
        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w

        force_expected_b = clamp_norm(
            self.force_kp_scaled * self.project_pos_diff(
                force_origin_b - force_keypoint_b,
                force_dir=force_dir_b,
            ),
            self.max_force * self.force_alpha
        )
        self.force_expected_b[:] = force_expected_b
        self.force_expected_w[:] = quat_apply(root_quat_exp, force_expected_b)

    # this function is for baseline (non-compliant perturbation)
    def force_update_perturb_and_target(self):
        self.force_sample_timer -= 1
        time_done = self.force_sample_timer <= 0
        need_resample = time_done.nonzero(as_tuple=False).squeeze(-1)
        if need_resample.numel() > 0:
            transit_time = random_uniform(need_resample.shape[0], 20, 50, self.device).int()
            hold_time = random_uniform(need_resample.shape[0], 20, 100, self.device).int()
            self.force_sample_timer[need_resample] = transit_time + hold_time

            force = rand_points_isotropic(
                need_resample.shape[0],
                2,
                r_max=float(self.max_force) * float(self.force_alpha),
                device=self.device,
            )  # [K, 2, 3]
            force_enable = (torch.rand(need_resample.shape[0], 2, device=self.device) < 0.5).unsqueeze(-1)  # [K, 2, 1]
            self.perturb_force.set(need_resample, end=force * force_enable, total_steps=transit_time)

        # force target: track the reference keypoints (no admittance)
        self.force_keypoint_w[:] = self.reward_keypoints_w[:, self.force_apply_idx_motion]

        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat_w = self.asset.data.root_quat_w.unsqueeze(1)
        self.force_keypoint_b[:] = quat_apply_inverse(root_quat_w, self.force_keypoint_w - root_pos_w)

        if self.last_reset_env_ids is not None:
            self.force_keypoint_w_prev[self.last_reset_env_ids] = self.force_keypoint_w[self.last_reset_env_ids]
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        self.force_keypoint_w_prev[:] = self.force_keypoint_w

    def project_pos_diff(self, pos_diff: torch.Tensor, force_dir: torch.Tensor):
        coef = (pos_diff * force_dir).sum(dim=-1, keepdim=True).clamp_max(0.0)
        return coef * force_dir
    
    def project_vel(self, vel: torch.Tensor, force_dir: torch.Tensor):
        coef = (vel * force_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)
        return coef * force_dir

    def _limit_net_wrench_about_torso(self):
        apply_idx = self.force_apply_idx_asset                     # [M]
        torso_i   = self.torso_idx_asset

        pos_w_6   = self.asset.data.body_pos_w[:, apply_idx, :]    # [N, M, 3]
        F_w_6     = self.force_apply_buffer[:, apply_idx, :]       # [N, M, 3]
        Tau_w_6   = self.torque_apply_buffer[:, apply_idx, :]      # [N, M, 3]

        p_torso   = self.asset.data.body_pos_w[:, self.torso_idx_asset, :].unsqueeze(1)  # [N,1,3]
        r_w_6     = pos_w_6 - p_torso                                                    # [N, M, 3]

        F_net = F_w_6.sum(dim=1)                                     # [N,3]
        M_net = torch.cross(r_w_6, F_w_6, dim=-1).sum(dim=1) + Tau_w_6.sum(dim=1)  # [N,3]

        eps = 1e-8
        
        F_norm   = F_net.norm(dim=-1, keepdim=True).clamp_min(eps)   # [N,1]
        F_scale  = torch.clamp(self.net_force_limit / F_norm, max=1.0)
        F_allow  = F_net * F_scale                                   # [N,3]
        dF       = F_allow - F_net                                   # [N,3]
        
        M_norm   = M_net.norm(dim=-1, keepdim=True).clamp_min(eps)   # [N,1]
        M_scale  = torch.clamp(self.net_torque_limit / M_norm, max=1.0)
        M_allow  = M_net * M_scale
        dM       = M_allow - M_net

        self.force_apply_buffer[:, torso_i,  :] = dF
        self.torque_apply_buffer[:, torso_i, :] = dM

    def force_apply(self, substep):
        if substep == 0:
            self.force_applied_w.zero_()
            self.force_applied_b.zero_()
            self.force_apply_buffer.zero_()

        pos_w = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
        quat_w = self.asset.data.body_quat_w[:, self.force_apply_idx_asset, :]

        self.force_applied_w[:] = clamp_norm(self.force_kp_scaled * self.project_pos_diff(self.force_origin_w - pos_w, force_dir=self.force_dir_w), self.max_force * self.force_alpha)
        self.force_applied_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.force_applied_w
        )
        self.force_apply_buffer[:, self.force_apply_idx_asset, :] = self.force_applied_w

        delta_w = quat_apply(quat_w, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset, :] = torch.cross(delta_w, self.force_applied_w, dim=-1)

        # override original force apply
        self.asset.has_external_wrench = False
        self.force_apply_world = True

    # this function is for baseline (non-compliant perturbation)
    def force_apply_perturb(self, substep: int):
        quat_w = self.asset.data.body_quat_w[:, self.force_apply_idx_asset, :]
        self.force_applied_w[:, -2:] = self.perturb_force.current  # [N, 2, 3]
        self.force_applied_b[:] = quat_apply_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.force_applied_w,
        )
        self.force_apply_buffer[:, self.force_apply_idx_asset, :] = self.force_applied_w

        delta_w = quat_apply(quat_w, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset, :] = torch.cross(delta_w, self.force_applied_w, dim=-1)

        # override original force apply
        self.asset.has_external_wrench = False
        self.force_apply_world = True

    def before_update(self):
        super().before_update()
        if self.compliance:
            self.force_kp_tl.update_time()
            self.force_origin_tl.update_time()
            self.force_safe_limit_tl.update_time()
            self.force_schedule()
            self.force_update_origin_and_target()
        else:
            self.perturb_force.update_time()
            self.force_update_perturb_and_target()

    def step(self, substep: int):
        super().step(substep)
        if self.compliance:
            self.force_apply(substep)
        elif self.max_force > 0.0:
            self.force_apply_perturb(substep)
        self._limit_net_wrench_about_torso()
    
    def step_schedule(self, progress: float):
        self.zero_init_prob = 0.0
        self.force_alpha = 1.0
        if not self.student_train:
            ratio = max(0.25, min(progress / 0.6, 1.0))
            force_prob = 0.15 * ratio
            self.force_type_probs[1:5] = force_prob
            self.force_type_probs[0] = 1.0 - force_prob * 4

    @observation
    def command(self):
        # here we use root pos instead of student body pos
        root_yaw_quat = yaw_quat(self._motion.root_quat_w[:, 0, :]).unsqueeze(1)
        root_yaw_future = yaw_quat(self._motion.root_quat_w[:, 1:, :])
        root_pos = self._motion.root_pos_w[:, 0, :].unsqueeze(1)
        root_pos_future = self._motion.root_pos_w[:, 1:, :]

        pos_diff_b = quat_apply_inverse(
            root_yaw_quat,
            root_pos_future - root_pos
        )

        heading = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        target_heading = quat_apply(root_yaw_future, heading)
        target_heading_b = quat_apply_inverse(root_yaw_quat, target_heading)

        return torch.cat([
            self._motion.root_pos_w[:, :, 2].reshape(self.num_envs, -1),
            pos_diff_b[:, :, :2].reshape(self.num_envs, -1),
            target_heading_b[:, :, :2].reshape(self.num_envs, -1),
            self.force_safe_limit_tl.current
        ], dim=-1)

    def command_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]).repeat(len(self.future_steps)),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]).repeat(len(self.future_steps) - 1),
            sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1]).repeat(len(self.future_steps) - 1),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
        ])

    @observation
    def force_priv(self):
        return torch.cat([
            self.force_keypoint_b.reshape(self.num_envs, -1),
            self.force_applied_b.reshape(self.num_envs, -1),
            self.force_expected_b.reshape(self.num_envs, -1),
            self.force_sample_timer.reshape(self.num_envs, -1)
        ], dim=-1)
    
    def force_priv_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.cartesian_space_symmetry(self.asset,get_items_by_index(self.asset.body_names, self.force_apply_idx_asset), sign=[1, -1, 1]).repeat(3),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1])
        ])

    @reward
    def keypoint_tracking_imp(self):
        actual = self.asset.data.body_pos_w[:, self.keypoint_idx_asset]
        target = self.reward_keypoints_w[:, self.keypoint_idx_motion].clone()
        target[:, self.force_in_keypoint_idx] = self.force_keypoint_w
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, self.reward_sigma["keypoint"])

    @reward
    def force_target_tracking(self):
        actual = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
        target = self.force_keypoint_w
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["force_target"])

    @reward
    def force_target_vel_tracking(self):
        actual_vel = self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset, :]
        target_vel = self.force_keypoint_vel_w
        diff = target_vel - actual_vel
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, self.reward_sigma["force_target_vel"])

    @reward
    def force_reward(self):
        force_norm = self.force_applied_w.norm(dim=-1, keepdim=False)
        force_norm_diff = (self.force_applied_w - self.force_expected_w).norm(dim=-1).mean(dim=-1, keepdim=True)  # [N, M]
        force_reward = _calc_exp_sigma(force_norm_diff, self.reward_sigma["force"])
        force_max_limit = self.force_safe_limit_tl.current + self.force_penalty_offset
        force_exd = (force_norm > force_max_limit).any(dim=-1, keepdim=True)
        return (force_reward * (~force_exd)).mean(dim=-1, keepdim=True)

    @reward
    def force_exd_penalty(self):
        force_norm = self.force_applied_w.norm(dim=-1, keepdim=False)
        force_exp_norm = self.force_expected_w.norm(dim=-1, keepdim=False)
        force_max_limit = self.force_safe_limit_tl.current + self.force_penalty_offset
        force_exd = ((force_norm > force_max_limit) & (force_norm > force_exp_norm + self.force_penalty_offset*0.5)).float().mean(dim=-1, keepdim=True) # [N, 1]
        return - force_exd

    def debug_draw(self):
        super().debug_draw()
        pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :]
        force = self.force_applied_w.clone()
        pos_flat   = pos.reshape(-1, 3)
        force_flat = force.reshape(-1, 3)

        self.env.debug_draw.vector(
            pos_flat, force_flat * 0.2,
            color=(0.0, 0.0, 1.0, 1.0)
        )

        act1 = self.asset.data.body_pos_w[:, self.force_apply_idx_asset, :].reshape(-1, 3)
        tar1 = self.force_keypoint_w.reshape(-1, 3)
        self.env.debug_draw.vector(
            act1,
            (tar1 - act1).reshape(-1, 3),
            color=(1, 0, 1, 1)
        )
        self.env.debug_draw.point(
            tar1.reshape(-1, 3), color=(1, 0, 1, 1), size=10.0
        )
        self.env.debug_draw.point(
            act1.reshape(-1, 3), color=(0, 0, 1, 1), size=10.0
        )

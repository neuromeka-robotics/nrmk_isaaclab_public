from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (  # noqa: F401
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    wrap_to_pi,
)

from isaac_neuromeka.assets.articulation import FiniteArticulation


class FloatingBaseVelocityAction(ActionTerm):
    """
    Floating-base 스타일의 로봇 제어를 위한 ActionTerm 클래스.
    Vx (전진 속도), Wz (Yaw 회전 속도) 만을 사용해 베이스를 직접 이동시킴.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cfg = cfg
        self._robot: FiniteArticulation = env.scene.articulations["robot"]

        self.total_mass = self._robot.root_physx_view.get_masses()[0].sum().item()

        print("MASS", self._robot.root_physx_view.get_masses()[0])
        print("body_names", self._robot.body_names)
        print("COM", self._robot.root_physx_view.get_coms()[0].shape)

        self._raw_actions = torch.zeros((env.num_envs, 2), device=env.device)  # [Vx, steer_angle]

        # computed from actions
        self.desired_velocity = torch.zeros((env.num_envs, 6), device=env.device)

        # 이전 속도 저장용
        self.prev_desired_velocity = torch.zeros((env.num_envs, 6), device=env.device)

        # Parameters
        self.dt = env.physics_dt

        self.front_idx = self.cfg.front_idx
        self.velocity_scale = self.cfg.velocity_scale
        self.yaw_rate_scale = self.cfg.yaw_rate_scale
        self.max_linear_acc = self.cfg.max_linear_accel * self.dt
        self.max_angular_acc = self.cfg.max_angular_accel * self.dt
        self.max_front_velocity = self.cfg.max_front_velocity
        self.max_yaw_rate = self.cfg.max_yaw_rate

        self.fixed_z_pose = self.cfg.fixed_z_pos

        self.z_kp = self.cfg.z_kp
        self.angle_kp = self.cfg.angle_kp

        self.z_kd = self.cfg.z_kd
        self.angle_kd = self.cfg.angle_kd

        self.xy_kp = self.cfg.xy_kp
        self.xy_kd = self.cfg.xy_kd

        self.yaw_kp = self.cfg.yaw_kp
        self.yaw_kd = self.cfg.yaw_kd

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_term_dim(self) -> int:
        return self.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.cat(
            [self.desired_velocity[:, self.front_idx].unsqueeze(1), self.desired_velocity[:, 5].unsqueeze(1)], dim=-1
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.prev_desired_velocity[env_ids] = 0.0
        self.desired_velocity[env_ids] = 0.0

        root_state = self._robot.data.root_state_w[env_ids].clone()
        # target_vel = torch.zeros_like(robot_velocity_w)
        target_vel = root_state[:, 7:13].clone() * 0.0
        self._robot.write_root_velocity_to_sim(target_vel, env_ids)

    # Runs at control dt
    def process_actions(self, actions: torch.Tensor):

        target_front_vel = actions[:, 0] * self.velocity_scale
        target_yaw_rate = actions[:, 1] * self.yaw_rate_scale

        target_front_vel = torch.clamp(target_front_vel, -self.max_front_velocity, self.max_front_velocity)
        target_yaw_rate = torch.clamp(target_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)  # 최대 회전 속도 제한

        self.desired_velocity[:] = 0.0
        self.desired_velocity[:, self.front_idx] = target_front_vel
        self.desired_velocity[:, 5] = target_yaw_rate

    # Runs at physics dt
    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # # Robot_state
        robot_velocity_w = self._robot._finite_body_vel_w[env_ids, 0]  # shape (N, 6)

        robot_quat_w = self._robot.data.root_quat_w[env_ids]  # shape (N, 4)
        robot_velocity_b = quat_apply_inverse(
            robot_quat_w, robot_velocity_w[:, :3]
        )  # 로봇의 body frame에서의 속도 (N, 3)
        robot_angvel_b = quat_apply_inverse(
            robot_quat_w, robot_velocity_w[:, 3:6]
        )  # 로봇의 body frame에서의 각속도 (N, 3)

        v_f_des = self.desired_velocity[env_ids, self.front_idx]
        w_z_des = self.desired_velocity[env_ids, 5]

        # # Differential drive model
        euler_angles = euler_xyz_from_quat(robot_quat_w)

        # TODO: max speed limit
        root_state = self._robot.data.root_state_w[env_ids].clone()

        ext_force_w = torch.zeros_like(robot_velocity_w[:, :3])
        ext_force_w[:, 2] = self.z_kp * (self.fixed_z_pose - root_state[:, 2])  # TODO: combine terrain height.
        ext_force_w[:, 2] -= self.z_kd * robot_velocity_w[:, 2]

        ext_force_b = quat_apply_inverse(robot_quat_w, ext_force_w)  # [N, 3]
        ext_force_b[:, 0] = self.xy_kp * (v_f_des - robot_velocity_b[:, 0])  # X
        ext_force_b[:, 1] = self.xy_kp * (0.0 - robot_velocity_b[:, 1])  # Y
        ext_force_b[:, :2] -= self.xy_kd * robot_velocity_b[:, :2]

        ext_torque_b = torch.zeros_like(robot_velocity_w[:, 3:6])
        ## Simple implementation using euler angles
        ext_torque_b[:, 0] = self.angle_kp * (-euler_angles[0])  # roll
        ext_torque_b[:, 1] = self.angle_kp * (-euler_angles[1])  # pitch
        # ext_torque_b -= self.angle_damping * robot_angvel_b

        ext_torque_b[:, 0] -= self.angle_kd * robot_angvel_b[:, 0]
        ext_torque_b[:, 1] -= self.angle_kd * robot_angvel_b[:, 1]
        ext_torque_b[:, 2] = self.yaw_kp * (w_z_des - robot_angvel_b[:, 2])
        ext_torque_b[:, 2] -= self.yaw_kd * robot_angvel_b[:, 2]

        # self._robot.write_root_velocity_to_sim(target_vel, env_ids)
        self._robot.set_external_force_and_torque(
            ext_force_b.unsqueeze(1), ext_torque_b.unsqueeze(1), body_ids=[0], env_ids=env_ids, is_global=False
        )

        self._robot.set_joint_position_target(
            torch.zeros_like(self._robot._data.joint_pos_target[env_ids]), env_ids=env_ids
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_vel_visualizer"):
                print("Creating target velocity visualizer")
                self.target_vel_visualizer = VisualizationMarkers(self.cfg.target_vel_visualizer_cfg)
            self.target_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_vel_visualizer"):
                self.target_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        body_pos_w = self._robot.data.root_pos_w

        target_velocity_base = self.desired_velocity[:, :3]

        target_velocity_w = quat_apply(self._robot.data.root_quat_w, target_velocity_base)

        direction_vec = target_velocity_w / (torch.norm(target_velocity_w, dim=1, keepdim=True) + 1e-6)
        # compute quaternion from x-axis to direction_vec_normalized
        x_axis = (
            torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=body_pos_w.dtype)
            .unsqueeze(0)
            .repeat(direction_vec.shape[0], 1)
        )

        axis = torch.cross(x_axis, direction_vec, dim=-1)
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis_normalized = axis / (axis_norm + 1e-8)
        angle = torch.acos(torch.clamp(torch.sum(x_axis * direction_vec, dim=-1, keepdim=True), -1.0, 1.0))
        arrow_quat = quat_from_angle_axis(angle.flatten(), axis_normalized)

        arrow_base_pos = body_pos_w + torch.tensor([0.0, 0.0, 0.5], device=body_pos_w.device).unsqueeze(0).repeat(
            body_pos_w.shape[0], 1
        )

        scale = torch.ones_like(arrow_base_pos) * 0.5
        scale[:, 0] = torch.clamp(torch.norm(target_velocity_w, dim=1, keepdim=False), min=0.1, max=2.0)

        self.target_vel_visualizer.visualize(arrow_base_pos, arrow_quat, scale)

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import pdb
from collections.abc import Sequence
import warnings
import carb
from collections.abc import Callable

import numpy as np
import torch
# from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventManager,
    ObservationManager,
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager,
    RecorderManager
)

# custom cfg
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg, RLEnvWithIKCfg


from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

from isaac_neuromeka.env.managers import*


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # -- container for extra information
        self.extra_data = dict()
        super().__init__(cfg, render_mode, **kwargs)
        # -- container for delay randomization
        self.delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


    def _pre_observation_compute_step(self):
        return
    #     if self.pointcloud_manager is not None:
    #         self.pointcloud_manager.update(dt=self.step_dt, force_recompute=False)
               

    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action)
        # perform physics stepping
        for i in range(self.cfg.decimation):
            # set actions into buffers w/ delay
            execute_env_ids = torch.where(self.delay_steps <= i)[0]
            self.action_manager.apply_action(execute_env_ids)
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # perform rendering if gui is enabled
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        
        # -- cost computation
        self.cost_buf = self.cost_manager.compute(dt=self.step_dt)
        self.extras["costs"] = self.cost_buf

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        
                        
        # post-sim step
        self._pre_observation_compute_step()

        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def load_managers(self):
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        # call the parent class to load the managers for observations and actions.


        # prepare the managers
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = CustomActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = CustomObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = CustomRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)
        
        # CMDP specific
        self.cost_manager = CostManager(self.cfg.costs, self)
        print("[INFO] Cost Manager: ", self.cost_manager)


    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        cost_info = self.cost_manager.reset(env_ids)
        self.extras["log"].update(cost_info)
        
    def close(self):
        if not self._is_closed:
            del self.cost_manager
        super().close()
    




""" 
Environment with IK Solver 
"""


from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.utils.math import subtract_frame_transforms



# Only supports single body for now.
class RLEnvWithIK(CustomManagerBasedRLEnv):
    cfg: RLEnvWithIKCfg
    
    def __pre_manager_init__(self):
               
        self.ik_method = self.cfg.ik_method
        self.ik_body_name = self.cfg.ik_body_name
        self.ik_cmd_name = self.cfg.ik_cmd_name
        ik_params: dict[str, float] | None = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 1.0},
        }

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.ik_method,
            ik_params=ik_params,
        )

        self.ik_solver = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )

        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[".*"], body_names=[self.ik_body_name]
        )
        robot_entity_cfg.resolve(self.scene)

        self.robot = self.scene[robot_entity_cfg.name]
        self.ee_idx = self.robot.find_bodies(self.ik_body_name)[0][0]
        num_joints = self.robot.num_joints
        
        self.ik_solution = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.ik_solution_clip = torch.zeros((self.num_envs, num_joints), device=self.device)

        if self.robot.is_fixed_base:
            self.jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            self.jacobi_idx = robot_entity_cfg.body_ids[0]

        self.joint_pos_target = torch.zeros(
            (self.num_envs, num_joints), device=self.device
        )
        self.prev_joint_pos_target = torch.zeros(
            (self.num_envs, num_joints), device=self.device
        )
        
 

    def _update_ik(self):
        self.ik_solver.reset()

        command = self.command_manager.get_term(self.ik_cmd_name).pose_command_b
        self.ik_solver.set_command(command)

        # obtain quantities from simulation
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, :]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, :]

        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        # compute the joint commands
        self.ik_solution = self.ik_solver.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        
        # clip the ik solution
        diff = self.ik_solution - joint_pos
        diff = torch.clamp(diff, -0.1, 0.1)
        self.ik_solution_clip = joint_pos + diff

    def _pre_observation_compute_step(self):
        super()._pre_observation_compute_step()
        self._update_ik()
    
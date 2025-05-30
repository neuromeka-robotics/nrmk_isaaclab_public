
"""MEMO
1. collision body fix: isaac_neuromeka/assets/model/urdf/moby.urdf -> 인디 부분도
2. Action space fix (tract joint -> joint velocity action)
3. actuator fix in isaac_neuromeka/assets/moby.py -> needs to be fixed in the future
4. contact sensors have to be added
"""

from __future__ import annotations

import pdb
from dataclasses import MISSING

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaac_neuromeka.utils.etc import EmptyCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg

# Import common environment configuration
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaac_neuromeka.env.rl_task_custom_env import HistoryManager



from isaac_neuromeka.mdp.actions.joint_actions import ClampedJointPositionAction, JointVelocityAction
from isaac_neuromeka.mdp.actions.base_actions import MobyBaseAction
from isaac_neuromeka.mdp.actions.DOF4_actions import Moby4DoFAction

from isaac_neuromeka.mdp.actions.action_cfgs import ClampedJointActionCfg, MobyBaseActionCfg, Moby4DoFActionCfg
from isaac_neuromeka.assets.articulation import FiniteArticulationCfg

from isaac_neuromeka.assets import MOBY_CFG

# Custom MDP stuff
import isaac_neuromeka.tasks.navigation.moby.mdp as mdp
from isaac_neuromeka.terrains.mesh_terrain_cfg import MeshBoxTerrainCfg
from isaac_neuromeka.terrains.terrain_importer_no_overlap import TerrainImporterNoOverlap
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
##
# Scene definition
##
from isaac_neuromeka.terrains.mesh_terrain_cfg import MeshMountainTerrainCfg
from isaaclab.managers import CommandTerm
import torch
import math

@configclass
class MobySceneCfg(InteractiveSceneCfg):

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    
    # robots
    robot: FiniteArticulationCfg = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # robot: FiniteArticulationCfg = MOBY_CFG.replace(
    # prim_path="{ENV_REGEX_NS}/robot",
    # init_state=FiniteArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.6),  # 지형 위로 띄워서 배치
    #     joint_pos=MOBY_CFG.init_state.joint_pos
    # )
    # )

    
    # target object
    obstacle: RigidObjectCfg = None

    # contact sensor # TODO: add contact sensor (also enable in moby.py in isaac_neuromeka/assets)
    contact_sensors =  None # To be added in the future for base collision detection
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

##
# Environment configuration
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        
        ### Arm states
        # joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05))
        # joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.5))
        # joint_pos_history = ObsTerm(func=HistoryManager, params={"name": "joint_pos", "length": 2})
        # joint_vel_history = ObsTerm(func=HistoryManager, params={"name": "joint_vel", "length": 2})
        
        ### Base states
        # TODO: get date from _finite_body_vel_w for stability
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        ### COMMAND
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_pose"})

        ### ETC
        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            

    # observation groups
    policy: PolicyCfg  = PolicyCfg()
    
    
@configclass
class ActionsCfg:
    
# TODO: experiment different low-level action

    """Action specifications for the MDP."""
    # arm_action: ActionTerm = MISSING
    steer_action: ActionTerm = ClampedJointActionCfg( class_type=ClampedJointPositionAction,
                                                          asset_name="robot",
                                                          joint_names=[".*_rot_joint"], 
                                                          clamp_range=(-1.5, 1.5), # radian -1.5,1.5   1.309
                                                          scale=0.25, use_default_offset=False)
    
    tract_action: ActionTerm = mdp.JointVelocityActionCfg(class_type=JointVelocityAction,
                                                          asset_name="robot",
                                                          joint_names=[".*_tract_joint"], 
                                                          scale=0.5, use_default_offset=False)              
@configclass
class SimplifiedAction:
    """
    Use Vx, Vy, Wz to control the robot via custom Ackermann + Swerve controller
    """
    vel_action: ActionTerm = MobyBaseActionCfg(
        class_type=MobyBaseAction,
        asset_name="robot",
        joint_names=[
            "fl_rot_joint", "fr_rot_joint", "rl_rot_joint", "rr_rot_joint",
            "fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"
        ],
        scale=1.0,
        offset=0.0,
        preserve_order=True,
    )

@configclass
class DOF4Action:
    """
    Use Vx, Vy, Wz to control the robot via custom Ackermann + Swerve controller
    """
    dof4_action: ActionTerm = Moby4DoFActionCfg(
        class_type=Moby4DoFAction,
        asset_name="robot",
        joint_names=[
            "fl_rot_joint", "fr_rot_joint", "rl_rot_joint", "rr_rot_joint",
            "fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"
        ],
        scale=1.0,
        offset=0.0,
        preserve_order=True,
    )

@configclass
class CommandsCfg: 
    
    # CUSTOM COMMAND -> Needs debugging.
    # base_pose = mdp.Nav2DPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="base_footprint", # TODO: check the body name
    #     resampling_time_range=(5.0, 10.0),
    #     debug_vis=True,
    #     make_quat_unique=True,
    #     ranges=mdp.Nav2DPoseCommandCfg.Ranges( 
    #         pos_x=(-2.0, 2.0),
    #         pos_y=(-1.0, 1.0),
    #         yaw=( -3.14, 3.14), 
    #     ),
    # )   
    
    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(16.0, 16.1), #5.0,10.0 /9.9,10
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            pos_x=(-2.0, 2.0),
            pos_y=(-1.0, 1.0),
            # pos_x=(2.0, 2.1),
            # pos_y=(2.0, 2.1),
            #pos_z=(0.5, 0.5),
            heading=( -3.14, 3.14), 
        ),
    )
    

    
    

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # action penalty
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)  #-0.0001
    action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)  # -0.00005, -0.0001

    # joint_vel = RewTerm(
    #     func=mdp.finite_joint_vel_l2,
    #     weight=-0.0005,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

 
    base_position_tracking = RewTerm(
        func=mdp.position_command_tracking_bounded, # Bound the distance for stability
        weight= 1.0, #1.0
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
                "command_name": "base_pose",
                "max_distance": 3.0},
    )
    
    body_orientation_tracking = RewTerm(
        func=mdp.yaw_tracking,
        weight= 0.2,      #0.1 0.15  4.20까지-0.2
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
                "command_name": "base_pose",
                "distance_max": 0.5}, # track only within 0.75 m 
    )

    acceleration_penalty = RewTerm(
        func=mdp.acceleration_penalty,
        weight= 0.15,   #0.15
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"), "command_name": "base_pose"}
    )
#     wheel_slip_penalty = RewTerm(
#     func=mdp.wheel_slip_penalty, 
#     weight=-0.1,
#     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_footprint")}
# )
    wheel_slip_penalty = RewTerm(
    func=mdp.wheel_slip_penalty,
    weight=-0.00001,
    params={"asset_cfg": SceneEntityCfg("robot")}
    )

    lateral_slip_penalty = RewTerm(
    func=mdp.lateral_slip_penalty,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # near_goal_stability = RewTerm(
    # func=mdp.near_goal_stability_reward,
    # weight=1.0,
    # params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
    #     "command_name": "base_pose",
    #     "distance_threshold": 0.2,
    #     "yaw_threshold_rad": math.radians(5.0),
    # }
    # )

    # steering_alignment = RewTerm(
    # func=mdp.steering_zero_alignment_reward,
    # weight=1.0,
    # params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
    #     "command_name": "base_pose",
    #     "distance_threshold": 0.2,
    #     "yaw_threshold_rad": math.radians(5.0),
    #     "gamma": 1.0,
    # }
    # )

    stop_velocity_reward_term = RewTerm(
    func=mdp.stop_velocity_reward,
    weight=1.0,  # 리워드의 중요도를 설정하는 weight 값
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),  # 로봇 설정
        "command_name": "base_pose",  # 목표 이름
        "distance_threshold": 0.15,  # 목표 도달 오차 범위 0.15
        "yaw_threshold_rad": math.radians(3.0),  # 목표 yaw 오차 범위 3.0
        "gamma": 1.0,  # 속도 보상에 대한 gamma 값
    }
    )



#     velocity_alignment = RewTerm(
#     func=mdp.velocity_direction_alignment_reward,
#     weight=1.0,
#     params={
#         "asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
#         "command_name": "base_pose",
#         "distance_threshold": 1.2,
#         "gamma": 5.0,
#     }
# )



    # angle_smoothness_penalty = RewTerm(
    #     func=mdp.angle_smoothness_penalty,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )


    # velocity_smoothness_penalty = RewTerm(
    #     func=mdp.velocity_smoothness_penalty,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # Trick for faster learning 
    # velocity_bonus = RewTerm(
    #     func=mdp.velocity_bonus,
    #     weight= 0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_footprint"),
    #             "command_name": "base_pose"},
    # )
    
    #TODO: Add reward for.. (1) wheel slippage (2) acceleration penalty  (3) too big action for wheel vel.
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# @configclass
# class SimplifiedAction:
    
# # TODO: experiment different low-level action

#     vel_action: ActionTerm = ClampedJointActionCfg( class_type=ClampedJointPositionAction,
#                                                           asset_name="robot",
#                                                           joint_names=[".*_rot_joint"], 
#                                                           clamp_range=(-1.5, 1.5), # radian
#                                                           scale=0.25, use_default_offset=False)



@configclass
class MobyEnvCfg(NrmkRLEnvCfg): 
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: MobySceneCfg = MobySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    #actions: ActionsCfg = ActionsCfg()
    actions: SimplifiedAction = SimplifiedAction()
    #actions: DOF4Action = DOF4Action()

    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg | EmptyCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:  EmptyCfg = EmptyCfg() # TODO: Later 
    curriculum = EmptyCfg() # Not used for now
    # CMDP settings
    costs: EmptyCfg = EmptyCfg() # Not used for now # TODO: Later 
    
    # 
    actor_obs_list: list = ["policy"] # ["proprioception", "point_cloud", "privileged"]
    critic_obs_list: list | None = None # None: same as actor_obs_list
    teacher_obs_list: list | None = None # None: same as actor_obs_list


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # task settings
        self.sim.dt = 1.0 / 120.0
        self.decimation = 24  # 24 * 1/120 = 0.2s
        self.episode_length_s = 100.0  #10 100
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)


@configclass
class TestCommandsCfg: 
    
    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            # pos_x=(3.0, 7.1),
            # pos_y=(3.0, 7.1),
            # #pos_z=(0.5, 0.5),
            # heading=( -3.14, 3.14),     
            # pos_x=(5.0, 9.1),
            # pos_y=(5.0, 9.1),
            pos_x=(5.0, 5.3),
            pos_y=(5.0, 5.0),
            heading=( -3.14, 3.14),      
  
        ),
    )


@configclass
class TestCommandsCfg1:

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            pos_x=(6.0, 6.0),
            pos_y=(6.0, 6.0),
            heading=(3.14, 3.14), 
        ),
    )

@configclass
class TestCommandsCfg2:

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            pos_x=(6.0, 6.0),
            pos_y=(6.0, 6.0),
            heading=(3.14, 3.14), 
        ),
    )

@configclass
class TestCommandsCfg3:

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            pos_x=(-2.0, -2.0),
            pos_y=(-2.0, -2.0),
            heading=(-3.14, -3.14), 
        ),
    )

@configclass
class TestCommandsCfg4:

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges( 
            pos_x=(2.0, 2.0),
            pos_y=(2.0, 2.0),
            heading=(-3.14, -3.14), 
        ),
    )


    


@configclass
class MobyTestEnvCfg(MobyEnvCfg):

    commands: CommandsCfg = TestCommandsCfg()
    

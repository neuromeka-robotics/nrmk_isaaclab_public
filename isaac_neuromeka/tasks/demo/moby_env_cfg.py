
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
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg # TODO: move one level up
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaac_neuromeka.env.rl_task_custom_env import HistoryManager


import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.mdp.actions import ClampedJointPositionAction, JointVelocityAction, CustomJointPositionAction
from isaac_neuromeka.mdp.actions.action_cfgs import ClampedJointActionCfg

from isaac_neuromeka.assets.articulation import FiniteArticulationCfg

from isaac_neuromeka.assets import MOBY_CFG
from isaaclab.terrains import TerrainImporterCfg

from isaac_neuromeka.terrain.mesh_terrain_cfg import MeshTerrainImporterCfg 


##
# Scene definition
##
from isaaclab.managers import CommandTerm
import torch
import math

# Test camera
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
import isaaclab.sim as sim_utils



@configclass
class MobySceneCfg(InteractiveSceneCfg):

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    # mesh = TerrainImporterCfg(
    #     prim_path="/World/mesh",
    #     terrain_type="usd",
    #     usd_path="/home/nrmk/Documents/usd_test/mesh_test.usd",
    # )
    
    # terrain = MeshTerrainImporterCfg(
    #     prim_path="/World/mesh",
    #     obj_dir = "isaac_neuromeka/assets/terrain_meshes/demo0"
    # )
    

    # robots
    robot: FiniteArticulationCfg = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # target object
    obstacle: RigidObjectCfg = None

    # contact sensor # TODO: add contact sensor (also enable in moby.py in isaac_neuromeka/assets)
    contact_sensors =  None # To be added in the future for base collision detection
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    
    camera_front = TiledCameraCfg(
        prim_path="/World/envs/env_.*/robot/base_footprint/camera_front", # TODO: adjust the pose
        offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types = ["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
    )
    
##
# Environment configuration
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RobotStatesCfg(ObsGroup):
        q = ObsTerm(func=mdp.joint_pos)
        qdot = ObsTerm(func=mdp.finite_joint_vel)
        p = ObsTerm(func=mdp.body_pose_b, params={"body_name": "link6"}) # TODO: tcp
        pdot = ObsTerm(func=mdp.body_vel_b, params={"body_name": "link6"})
        op_state = ObsTerm(func=mdp.op_state)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05)) 
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        image = ObsTerm(func=mdp.image_unnormalized, params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "rgb"})
        depth_image = ObsTerm(func=mdp.image_unnormalized, params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "depth"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False


    # observation groups
    policy = RobotStatesCfg()

    
@configclass
class ActionsCfg:

    """Action specifications for the MDP."""
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["joint[0-5]"], scale=1.0, use_default_offset=False
        )
    steer_action: ActionTerm = ClampedJointActionCfg( class_type=ClampedJointPositionAction,
                                                          asset_name="robot",
                                                          joint_names=[".*_rot_joint"], 
                                                          clamp_range=(-1.5, 1.5), # radian -1.5,1.5   1.309
                                                          scale=1.0, use_default_offset=False)
    
    tract_action: ActionTerm = mdp.JointVelocityActionCfg(class_type=JointVelocityAction,
                                                          asset_name="robot",
                                                          joint_names=[".*_tract_joint"], 
                                                          scale=1.0, use_default_offset=False)              


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot_pose = EventTerm(
    #     func=mdp.reset_pose_mesh_terrain,
    #     mode="reset",
    #     params={
    #         "pose_range":  {"yaw": (0.0, 3.14)},
    #         "velocity_range":  {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     },
    # )

    # TODO: fix them
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint.*"),
            "friction_distribution_params": (0.7, 1.3),
            "armature_distribution_params": (0.75, 1.25),
            "operation": "abs",
            "distribution": "uniform"
        }
    )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class MobyDeployEnvCfg(NrmkRLEnvCfg): 
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: MobySceneCfg = MobySceneCfg(num_envs=1, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands = EmptyCfg()
    actions = ActionsCfg()
    rewards = EmptyCfg()
    curriculum = EmptyCfg() # Not used for now
    costs = EmptyCfg() # Not used for now
    terminations = TerminationsCfg() # Not used for now
    events = EventCfg()
    
    actor_obs_list = ["policy"]


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # task settings
        self.sim.dt = 1.0 / 120.0
        self.decimation = 24  # 24 * 1/120 = 0.2s
        self.episode_length_s = 100.

        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)



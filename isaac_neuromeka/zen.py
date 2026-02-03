import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaac_neuromeka.assets.articulation import (
    FiniteArticulation,
    FiniteArticulationCfg,
)

##
# Configuration
##

ZEN_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/paxini/paxini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            # Wheel joints
            "r_wheel_joint": 0.0,
            "l_wheel_joint": 0.0,
            # Rotation joint
            "rotate_joint": 0.0,
            # Waist joints
            "waist_1_joint": 0.0,
            "waist_2_joint": 0.0,
            # Neck and head joints
            "neck_joint": 0.0,
            "head_joint": 0.0,
            # Right arm joints
            "rarm_1_joint": 0.0,
            "rarm_2_joint": 0.0,
            "rarm_3_joint": 0.0,
            "rarm_4_joint": 0.0,
            "rarm_5_joint": 0.0,
            "rarm_6_joint": 0.0,
            # Left arm joints
            "larm_1_joint": 0.0,
            "larm_2_joint": 0.0,
            "larm_3_joint": 0.0,
            "larm_4_joint": 0.0,
            "larm_5_joint": 0.0,
            "larm_6_joint": 0.0,
        },
    ),
    actuators={
        # Wheel joints - velocity control
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["r_wheel_joint", "l_wheel_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=50.0,
            stiffness=0.0,  # velocity control only
            damping=20.0,
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Rotation joint
        "rotation_joint": ImplicitActuatorCfg(
            joint_names_expr=["rotate_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=100.0,
            stiffness=100.0,  # 100
            damping=20.0,  # 20
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Waist joints - Fixed/disabled to prevent movement
        "waist_joints": ImplicitActuatorCfg(
            joint_names_expr=["waist_1_joint", "waist_2_joint"],
            velocity_limit_sim=2.0,
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,
            # damping=2000.0,
            effort_limit_sim=100.0,
            stiffness=100.0,
            damping=20.0,
        ),
        # Neck and head joints
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["neck_joint", "head_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=100.0,
            stiffness=100.0,
            damping=20.0,
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Right arm joints - Indy arm0 → Paxini rarm_1_joint
        "rarm_0": ImplicitActuatorCfg(
            joint_names_expr=["rarm_1_joint"],
            velocity_limit_sim=2.775073510670984,  # Indy arm0
            effort_limit_sim=431.97,  # Indy arm0
            stiffness=100.0,
            damping=20.0,  # Indy arm0
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Right arm joints - Indy arm1 → Paxini rarm_2_joint
        "rarm_1": ImplicitActuatorCfg(
            joint_names_expr=["rarm_2_joint"],
            velocity_limit_sim=2.775073510670984,  # Indy arm1
            effort_limit_sim=197.23,  # Indy arm1
            stiffness=100.0,
            damping=20.0,  # Indy arm1
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Right arm joints - Indy arm2 → Paxini rarm_[3-6]_joint
        "rarm_2": ImplicitActuatorCfg(
            joint_names_expr=["rarm_[3-6]_joint"],
            velocity_limit_sim=3.2986722862692828,  # Indy arm2
            effort_limit_sim=79.79,  # Indy arm2
            stiffness=100.0,
            damping=20.0,  # Indy arm2
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Left arm joints - Indy arm0 → Paxini larm_1_joint
        "larm_0": ImplicitActuatorCfg(
            joint_names_expr=["larm_1_joint"],
            velocity_limit_sim=2.775073510670984,  # Indy arm0
            effort_limit_sim=431.97,  # Indy arm0
            stiffness=100.0,
            damping=20.0,  # Indy arm0
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Left arm joints - Indy arm1 → Paxini larm_2_joint
        "larm_1": ImplicitActuatorCfg(
            joint_names_expr=["larm_2_joint"],
            velocity_limit_sim=2.775073510670984,  # Indy arm1
            effort_limit_sim=197.23,  # Indy arm1
            stiffness=100.0,
            damping=20.0,  # Indy arm1
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Left arm joints - Indy arm2 → Paxini larm_[3-6]_joint
        "larm_2": ImplicitActuatorCfg(
            joint_names_expr=["larm_[3-6]_joint"],
            velocity_limit_sim=3.2986722862692828,  # Indy arm2
            effort_limit_sim=79.79,  # Indy arm2
            stiffness=100.0,
            damping=20.0,  # Indy arm2
            # effort_limit_sim=100000.0,
            # stiffness=10000000.0,  #100
            # damping=2000.0,  #20
        ),
        # Hand joints - right hand (position lock at 0.0)
    },
    soft_joint_pos_limit_factor=0.95,
)

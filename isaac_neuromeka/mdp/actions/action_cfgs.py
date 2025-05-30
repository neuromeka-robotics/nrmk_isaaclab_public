from isaaclab.utils import configclass

from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg

from isaac_neuromeka.mdp.actions.joint_actions import JointResidualAction, ClampedJointPositionAction

@configclass
class ResidualJointActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointResidualAction

    offset: float | dict[str, float] = 0.0  
    
    cmd_name: str = "ee_pose" 
    
    use_default_offset=True
    
    # for experiment
    repulsive_force_coeff: float = 0.1

@configclass
class ClampedJointActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = ClampedJointPositionAction

    offset: float | dict[str, float] = 0.0  
    
    clamp_range: tuple[float, float] = (-1.0, 1.0)
    
    cmd_name: str = "ee_pose" 
    
    use_default_offset=True
    




# @configclass
# class IKResidualActionCfg(ResidualJointActionCfg):
#     ik_method: str = "dls"
#     ik_body_name: str = "tcp"


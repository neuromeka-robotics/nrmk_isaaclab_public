import numpy as np
import torch

## Kinematics
import pytorch_kinematics as pk
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms.rotation_conversions import quaternion_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply, quaternion_invert

class SimpleIKSolver:
    def __init__(self, urdf_path, device="cuda"):
        self.target_ee_pose = None
        self.kin_chain = pk.build_chain_from_urdf( open(urdf_path).read())
        # self.robot_state = {}
        self.device = device


    # def set_state(self, robot_state:dict):
    #     self.robot_state = robot_state


    def solve(self, joint_pos:torch.Tensor, target_ee_pos:torch.Tensor, step_size=0.2, iterations = 5, damping=0.1):
        # joint_pos = torch.from_numpy(self.robot_state["q"]).to(device=self.device, dtype=torch.float32)
        
        new_joint_pos = joint_pos.clone()
        tcp_chain = pk.SerialChain(self.kin_chain, "tcp").to(device=self.device)
        target_rot = quaternion_to_matrix(target_ee_pos[3:7])
        
        for i in range(iterations):
            
            ee_pos: Transform3d = tcp_chain.forward_kinematics(new_joint_pos, end_only=True)
            
            pos_error = target_ee_pos[:3] - ee_pos.get_matrix()[:,:3, 3].flatten()
        
            ee_rot_b_inv = ee_pos._get_matrix_inverse()[:,:3, :3].squeeze()
            rotation_error = target_rot @ ee_rot_b_inv
            quat_error = matrix_to_quaternion(rotation_error)
            axis_angle_error = quaternion_to_axis_angle(quat_error) 
            
            delta_task_space = torch.cat([pos_error, axis_angle_error], dim=-1).to(dtype=torch.float32)
            jacobian = tcp_chain.jacobian(th=new_joint_pos).to(dtype=torch.float32)
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (damping**2) * torch.eye(n=jacobian.shape[1], device=delta_task_space.device, dtype=torch.float32)
            
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_task_space.unsqueeze(-1)
                )
            
            new_joint_pos = new_joint_pos + delta_joint_pos.flatten() * step_size        
    
        return new_joint_pos
    
    
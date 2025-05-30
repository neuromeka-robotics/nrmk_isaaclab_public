from __future__ import annotations
import argparse
import gymnasium as gym
import math
import numpy as np
import os
import pdb
import sys
import time
from datetime import datetime
import yaml

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with NRMK-RL.")

parser.add_argument("--debug_vis", action="store_true", default=False, help="Usage during real robot experiments.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")

"""Launch Isaac Sim Simulator first."""
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaac_neuromeka.tasks  # noqa: F401

# EnvWrapper
from env_wrapper.env_wrapper_base import EnvWrapper

# Communication
from zmq_wrapper.broadcast import ZmqPublisher, ZmqSubscriber
import os

# Low-level control
from controllers.simple_ik import SimpleIKSolver

# # ETC
# from test_utils.command import KeyboardPoseCommand



# subscribes task space target pose and publishes state information


def main():
    import importlib.resources as resources
    
    # Load the YAML configuration file
    parent_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(parent_path, "config", "indy_sim.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)


    communication_config = config["communication"]
    sim_ip = communication_config["sim_ip"]
    ports = communication_config["ports"]

    print(f"Communication config: {communication_config}")

    # IsaacLab task
    task = config["isaaclab_task"]

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=env_cfg)
    env = EnvWrapper(env, debug_vis=args_cli.debug_vis)
    obs, infos = env.reset()

    # debug_info = {
    #     "point_cloud": torch.zeros((1, 3), dtype=torch.float32),
    #     "voxels": torch.zeros((1, 3), dtype=torch.float32),
    #     "point_cloud2": torch.zeros((1, 3), dtype=torch.float32),
    # }
    # def debug_vis_callback(debug_msg):
    #     nonlocal debug_info
    #     if debug_msg is not None:
    #         debug_info = debug_msg


    # DEFINE PUB & SUB CLIENTS
    state_pub = ZmqPublisher(ip=sim_ip, port=ports["state"])

    task_space_cmd = torch.Tensor([0.3563, -0.1829, 0.5132, 0.0, 0.0, 1.0, 0.0])  # [x, y, z, qw, qx, qy, qz]
    task_space_cmd = task_space_cmd.to(env.device, dtype=torch.float32) 

    def cmd_callback(control_msg):
        nonlocal task_space_cmd
        if control_msg is not None:
            task_space_cmd = torch.from_numpy(control_msg).to(env.device)
        
    # def command_callback(command_msg):
    #     nonlocal target_ee_pose
    #     if command_msg is not None:
    #         target_ee_pose = command_msg

    cmd_sub = ZmqSubscriber(ip=sim_ip, port=ports["control"]).async_start(cmd_callback)
    # debug_msg_listener = ZmqSubscriber(ip=sim_ip, port=8890).async_start(debug_vis_callback)


    
    # Low-level control
    urdf_path = resources.files("isaac_neuromeka.assets").joinpath("model", "urdf", "indy7_simplified.urdf")
    ik_solver = SimpleIKSolver(urdf_path, device=env.device)
    joint_cmd = torch.zeros((1, env.num_actions), dtype=torch.float32)

    # Start simulation
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            start = time.time()



            # step simulation
            obs, _, _, infos = env.step(joint_cmd)            

            # obs to numpy
            obs_np = {k: v.cpu().numpy() for k, v in infos["observations"]["policy"].items()}
            state_pub.broadcast(obs_np)

            ik_solver.set_state(obs_np)
            joint_cmd = ik_solver.solve(target_ee_pos=task_space_cmd).reshape(env.num_envs, -1)


            # env.update_command(target_ee_pose)
            # env.update_debug_vis(point_cloud=debug_info["point_cloud"], voxels=debug_info["voxels"], point_cloud2=debug_info["point_cloud2"])

            wait_time = env.control_dt - (time.time() - start)
            if args_cli.real_time and wait_time > 0:
                time.sleep(wait_time)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main execution
    main()

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
parser = argparse.ArgumentParser(description="")

parser.add_argument("--debug_vis", action="store_true", default=False, help="Usage during real robot experiments.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")

"""Launch Isaac Sim Simulator first."""
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True) # (Override) Enable cameras by default
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
    yaml_path = os.path.join(parent_path, "config", "moby_sim.yaml")
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


    # DEFINE PUB & SUB CLIENTS
    state_pub = ZmqPublisher(ip=sim_ip, port=ports["state"])

    task_space_cmd = torch.Tensor([0.3563, -0.1829, 0.5132, 0.0, 0.0, 1.0, 0.0])  # [x, y, z, qw, qx, qy, qz]
    task_space_cmd = task_space_cmd.to(env.device, dtype=torch.float32) 

    def tcp_cmd_callback(control_msg):
        nonlocal task_space_cmd
        if control_msg is not None:
            task_space_cmd = torch.from_numpy(control_msg).to(env.device)
        

    tcp_cmd_sub = ZmqSubscriber(ip=sim_ip, port=ports["tcp_control"]).async_start(tcp_cmd_callback)

    
    # Low-level control
    urdf_path = resources.files("isaac_neuromeka.assets").joinpath("model", "urdf", "indy7_simplified.urdf")
    ik_solver = SimpleIKSolver(urdf_path, device=env.device)
    joint_cmd = torch.zeros((1, env.num_actions), dtype=torch.float32)

    # for image debugging
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 4))

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

            # IK for arm control
            arm_joint_pos = torch.from_numpy(obs_np["q"][:,:6]).to(device=ik_solver.device, dtype=torch.float32)
            arm_cmd = ik_solver.solve(joint_pos=arm_joint_pos,target_ee_pos=task_space_cmd).reshape(env.num_envs, -1)
            joint_cmd[:, :6] = arm_cmd

            # visualize images
            img = obs_np["image"][0].astype(np.float32) / 255.0
            depth_img = obs_np["depth_image"][0].astype(np.float32) 

            plt.figure(figure.number)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Published RGB Image")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(depth_img, cmap="gray")
            plt.title("Published Depth Image")
            plt.axis("off")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

            wait_time = env.control_dt - (time.time() - start)
            if args_cli.real_time and wait_time > 0:
                time.sleep(wait_time)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main execution
    main()


        # images = self.scene.sensors["camera_front"].data.output["rgb"]
        # images = images.float() / 255.0

        # depths = self.scene.sensors["camera_front"].data.output["depth"]
        # depths[torch.isnan(depths)] = 0
        # depths[torch.isinf(depths)] = 0



        # # visualize the first image in the batch using matplotlib (for debugging)


        # img = images[0].cpu().numpy()
        
        # depth_img = depths[0].cpu().numpy()
    
        # plt.subplot(1, 2, 1)
        # plt.figure(self.figure.number)
        # plt.clf()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.title("Camera RGB Image")
        # plt.axis("off")
        # plt.subplot(1, 2, 2)
        # plt.imshow(depth_img, cmap="gray")
        # plt.title("Camera Depth Image")
        # plt.axis("off")
        # plt.tight_layout()
        # plt.draw()
        # plt.pause(0.001)



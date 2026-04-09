"""
This script
- Runs the simulator
- Publishes sensor data (proprioception, exteroception, etc.)
- Subscribes to action commands and applies them to the robot in the simulator.

Pre-requisites:
- Install Isaac Sim (4.5.0), IsaacLab (v2.2.1), nrmk_isaaclab_public
    - Follow the README
- Install zenoh
        pip install eclipse-zenoh
        
How to use:
    python deploy/run_namy_sim.py
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="")

# [MUST] Launch Isaac Sim Simulator first
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)  # (Override) Enable cameras by default
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Other imports
import os
import time
import yaml
import numpy as np
import torch
import gymnasium as gym

# Communication
from deploy.utils.communication import ZenohBus

# EnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from env_wrapper.env_wrapper_base import EnvWrapper


def main():
    # Load the configuration file
    parent_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(parent_path, "config", "namy_sim.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Set communication
    communication_config = config["communication"]
    topic_namespace = communication_config.get("topic_namespace", "namy_sim")
    zenoh_bus = ZenohBus()

    # Create environment for the given IsaacLab task
    task = config["isaaclab_task"]
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=env_cfg)
    env = EnvWrapper(env)
    obs, infos = env.reset()

    # Set "action" subscriber
    action_buffer = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)  # [v_x, w_z ]

    def _action_command_handler(key: str, msg: bytes):
        arr = np.frombuffer(msg, dtype=np.float32)
        if arr.size < 2:
            return
        tensor = torch.from_numpy(arr).to(device=env.device, dtype=action_buffer.dtype)
        action_buffer[:, :2] = tensor

    zenoh_bus.subscribe(f"{topic_namespace}/action_command", _action_command_handler)

    # Start simulation
    while simulation_app.is_running():
        with torch.inference_mode():
            start = time.time()

            # Step simulation (Subscribed action will be applied here)
            obs, _, _, infos = env.step(action_buffer)

            # Publish states (proprioception, exteroception, etc.)
            obs_dict = infos["observations"]["policy"]
            for k, v in obs_dict.items():
                zenoh_bus.publish_torch_tensor(f"{topic_namespace}/obs/{k}", v)

            wait_time = env.control_dt - (time.time() - start)
            if wait_time > 0:
                time.sleep(wait_time)

    # Close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

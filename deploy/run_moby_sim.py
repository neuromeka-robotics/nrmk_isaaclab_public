# flake8: noqa F841
from __future__ import annotations

import argparse
import os
import pdb  # noqa:F401
import time

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="")

parser.add_argument("--debug_vis", action="store_true", default=False, help="Usage during real robot experiments.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")

"""Launch Isaac Sim Simulator first."""
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)  # (Override) Enable cameras by default
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
# Communication
# from zmq_wrapper.broadcast import ZmqPublisher, ZmqSubscriber
import os

import isaaclab_tasks  # noqa: F401
from deploy.utils.data import _to_uint8_image
from deploy.utils.communication import ZenohBus

# EnvWrapper
from env_wrapper.env_wrapper_base import EnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaac_neuromeka.tasks  # noqa: F401

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

    # -- Communication setup --
    communication_config = config["communication"]
    topic_namespace = communication_config.get("topic_namespace", "moby_sim")

    zenoh_bus = ZenohBus()

    # subscriber example
    def _array_handler(key: str, msg: bytes):
        if "meta" in key:
            print(f"Received meta: {key} : {msg.decode('utf-8')}")
        else:
            arr = np.frombuffer(msg, dtype=np.float32)
            tensor = torch.from_numpy(arr)
            print(f"Received tensor: {key} : {tensor}")

    zenoh_bus.subscribe(f"{topic_namespace}/obs/**", _array_handler)

    # IsaacLab task
    task = config["isaaclab_task"]

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(task, cfg=env_cfg)
    env = EnvWrapper(env, debug_vis=args_cli.debug_vis)
    obs, infos = env.reset()

    # for image debugging
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 4))

    action_buffer = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)  # [v_x, w_z ]
    action_buffer[:, 0] = 1.0
    action_buffer[:, 1] = 1.0

    # Start simulation
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            start = time.time()

            # step simulation
            obs, _, _, infos = env.step(action_buffer)

            # # obs to numpy
            # obs_np = {k: v.cpu().numpy() for k, v in infos["observations"]["policy"].items()}
            # # state_pub.broadcast(obs_np)

            # Publish states
            obs_dict = infos["observations"]["policy"]
            for k, v in obs_dict.items():
                # print(f"{k}: {v.shape}, {v.dtype}")
                zenoh_bus.publish_torch_tensor(f"{topic_namespace}/obs/{k}", v)

            if args_cli.debug_vis:
                # visualize images

                img = _to_uint8_image(obs_dict["image"][0]).cpu().numpy()
                # img = obs_np["image"][0].astype(np.float32) / 255.0
                depth_img = obs_dict["depth_image"][0].cpu().numpy()

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

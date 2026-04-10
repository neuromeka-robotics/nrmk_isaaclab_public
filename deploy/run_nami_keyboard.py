"""
This script
- Obtain action command from the keyboard controller and publishes it to the simulator.
- Subscribes to the camera topics published by the simulator and visualizes them for debugging.

Pre-requisites:
- Install zenoh, open3d, matplotlib, and pynput
        pip install eclipse-zenoh open3d matplotlib pynput

How to use:
    python deploy/run_nami_keyboard.py
    python deploy/run_nami_keyboard.py --debug_vis   # To visualize images and pointcloud
"""

from __future__ import annotations

import argparse
import os
from threading import Lock

import numpy as np
import yaml
import time
from pynput import keyboard
from pynput.keyboard import Key

# Commnuication
from deploy.utils.communication import ZenohBus


class NamiKeyboardController:
    """Tracks pressed keys and converts them into a base action command."""

    def __init__(self, linear_speed: float, yaw_speed: float) -> None:
        self.linear_speed = linear_speed
        self.yaw_speed = yaw_speed
        self._lock = Lock()
        self._pressed_keys: set[object] = set()
        self._running = True
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        
    def close(self) -> None:
        with self._lock:
            self._running = False
        self._listener.stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_command(self) -> np.ndarray:
        with self._lock:
            pressed_keys = set(self._pressed_keys)

        forward = 0.0
        yaw = 0.0

        if Key.up in pressed_keys:
            forward = self.linear_speed
        if Key.down in pressed_keys:
            forward = -self.linear_speed
        if Key.left in pressed_keys:
            yaw = self.yaw_speed
        if Key.right in pressed_keys:
            yaw = -self.yaw_speed

        return np.array([forward, yaw], dtype=np.float32)
    
    def _normalize_key(self, key: object) -> object:
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return key

    def _on_press(self, key: object) -> None:
        norm_key = self._normalize_key(key)
        with self._lock:
            self._pressed_keys.add(norm_key)

    def _on_release(self, key: object) -> bool | None:
        norm_key = self._normalize_key(key)
        with self._lock:
            self._pressed_keys.discard(norm_key)
            if norm_key == Key.esc:
                self._running = False
                return False
        return None


def main() -> None:
    # Add argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_vis", action="store_true", default=False, help="Visualize camera images and point cloud for debugging.")
    args = parser.parse_args()
    
    # Load the configuration file
    parent_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(parent_path, "config", "nami_sim.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Set communication
    topic_namespace = config["communication"].get("topic_namespace", "nami_sim")
    command_topic = f"{topic_namespace}/action_command"
    image_topic = f"{topic_namespace}/obs/image"
    depth_topic = f"{topic_namespace}/obs/depth_image"
    zenoh_bus = ZenohBus()
    
    # Set extra configration (camera parameters, robot command parameters, etc.)
    # Currently, the camera configuration is matched to realsense D435.
    CAMERA_CONFIG = {
        "width": 640,
        "height": 480,
        "intrinsics": {
            "fx": 604.8516,
            "fy": 604.3739,
            "cx": 321.95575,
            "cy": 238.7731
        },
        "clipping_range": (0.1, 10.0),
        "pcl_subsample_stride": 4
    }
    ROBOT_CMD_CONFIG = {
        "forward": 1,
        "yaw": 1
    }
    
    # Set controller that outputs action command
    # Currently, keyboard is used. (Arrows for +forward/-forward/+yaw/-yaw, Esc for quit)
    # In future, neural network or other fancy algorithms can be used.
    controller = NamiKeyboardController(
        linear_speed=ROBOT_CMD_CONFIG["forward"], yaw_speed=ROBOT_CMD_CONFIG["yaw"])
    
    # Set visualizer for debugging
    if args.debug_vis:
        from deploy.utils.data import SensorVisualizer
        visualizer = SensorVisualizer(**CAMERA_CONFIG)
        zenoh_bus.subscribe(image_topic, lambda key, payload: visualizer.update_image(payload))
        zenoh_bus.subscribe(depth_topic, lambda key, payload: visualizer.update_depth(payload))

    try:
        while controller.is_running():
            # Get command and publish
            command = controller.get_command()
            zenoh_bus.publish(command_topic, command.tobytes())
            
            # Visualize sensor data for debugging
            if args.debug_vis:
                visualizer.draw()
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish one final zero command to stop the robot cleanly.
        zenoh_bus.publish(command_topic, np.zeros(2, dtype=np.float32).tobytes())
        controller.close()
        if args.debug_vis:
            visualizer.close()
        zenoh_bus.close()


if __name__ == "__main__":
    main()

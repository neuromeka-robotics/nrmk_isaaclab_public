"""
This script
- Obtain action command from the keyboard controller and publishes it to the simulator.
- Subscribes to the camera topics published by the simulator and visualizes them for debugging.

Pre-requisites:
- Install zenoh, open3d, matplotlib, and pynput
        pip install eclipse-zenoh open3d matplotlib pynput

How to use:
    python deploy/run_nav_keyboard.py
    python deploy/run_nav_keyboard.py --config deploy/configs/moby.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

_DEPLOY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOY_DIR.parent
for _path in (str(_REPO_ROOT), str(_DEPLOY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key

from deploy.config import command_configs
from deploy.config import key as topic_key
from deploy.config import load_config, mapping_section, parse_action_slice
from deploy.zenoh_bus import ZenohBus

DEFAULT_CONFIG_PATH = _DEPLOY_DIR / "configs" / "nami_nav.yaml"


class NavKeyboardController:
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


def _select_nav_command(config: Mapping[str, Any]) -> Mapping[str, Any]:
    commands = command_configs(config)
    keyboard_cfg = mapping_section(config, "keyboard")
    requested_topic = keyboard_cfg.get("command_topic")
    if requested_topic is not None:
        for command in commands:
            if str(command.get("topic", "")) == requested_topic:
                return command
        raise ValueError(f"No command topic {requested_topic!r} found in deploy config.")

    if len(commands) == 1:
        return commands[0]

    for command in commands:
        start, end = parse_action_slice(command.get("action_slice"))
        if end - start == 2:
            return command

    raise ValueError("Deploy config has multiple commands; set keyboard.command_topic to the two-value nav command.")


def main() -> None:
    # Add argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Deploy YAML config. Defaults to deploy/configs/nami_nav.yaml.",
    )
    args = parser.parse_args()

    # Load the configuration file
    config, _ = load_config(args.config)
    keyboard_cfg = mapping_section(config, "keyboard")
    command_cfg = _select_nav_command(config)
    command_start, command_end = parse_action_slice(command_cfg.get("action_slice"))
    command_dim = command_end - command_start
    if command_dim != 2:
        raise ValueError(
            f"Keyboard nav controller publishes [forward, yaw], but {command_cfg.get('topic')!r} "
            f"expects {command_dim} values."
        )
    command_dtype = str(command_cfg.get("dtype", "float32"))
    if command_dtype != "float32":
        raise ValueError(f"Unsupported command dtype {command_dtype!r}; only 'float32' is supported.")

    # Set communication
    communication = mapping_section(config, "communication")
    topic_namespace = str(communication.get("topic_namespace", config.get("isaaclab_task", "sim")))
    zenoh_config = communication.get("zenoh")
    observations_cfg = mapping_section(config, "observations")
    obs_root = str(observations_cfg.get("root", "obs"))
    command_topic = topic_key(topic_namespace, str(command_cfg.get("topic", "action_command")))
    image_topic = topic_key(topic_namespace, obs_root, "policy", "image")
    depth_topic = topic_key(topic_namespace, obs_root, "policy", "depth_image")
    zenoh_bus = ZenohBus(zenoh_config if isinstance(zenoh_config, dict) else None)

    # Set extra configuration (camera parameters, robot command parameters, etc.)
    # Currently, the camera configuration is matched to realsense D435.
    CAMERA_CONFIG = {
        "width": 640,
        "height": 480,
        "intrinsics": {"fx": 604.8516, "fy": 604.3739, "cx": 321.95575, "cy": 238.7731},
        "clipping_range": (0.1, 10.0),
        "pcl_subsample_stride": 4,
    }
    # Set controller that outputs action command
    # Currently, keyboard is used. (Arrows for +forward/-forward/+yaw/-yaw, Esc for quit)
    # In future, neural network or other fancy algorithms can be used.
    controller = NavKeyboardController(
        linear_speed=float(keyboard_cfg.get("linear_speed", 1.0)),
        yaw_speed=float(keyboard_cfg.get("yaw_speed", 1.0)),
    )

    # Set visualizer for debugging
    debug_vis = bool(keyboard_cfg.get("debug_vis", False))
    if debug_vis:
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
            if debug_vis:
                visualizer.draw()
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish one final zero command to stop the robot cleanly.
        zenoh_bus.publish(command_topic, np.zeros(2, dtype=np.float32).tobytes())
        controller.close()
        if debug_vis:
            visualizer.close()
        zenoh_bus.close()


if __name__ == "__main__":
    main()

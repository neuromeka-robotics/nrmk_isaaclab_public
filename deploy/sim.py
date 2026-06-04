from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_DEPLOY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOY_DIR.parent
for _path in (str(_REPO_ROOT), str(_DEPLOY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


from isaaclab.app import AppLauncher
from ruamel.yaml import YAML


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a generic Neuromeka IsaacLab deploy simulator. The Isaac Sim visualizer is enabled by default."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to an explicit deploy YAML config.")
    parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to match the env control rate.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many environment steps.")
    AppLauncher.add_app_launcher_args(parser)
    _ensure_headless_flag(parser)
    parser.set_defaults(enable_cameras=True)
    return parser


def _ensure_headless_flag(parser: argparse.ArgumentParser) -> None:
    if "--headless" not in parser._option_string_actions:
        parser.add_argument(
            "--headless", action="store_true", default=False, help="Run without the Isaac Sim visualizer."
        )
    parser.set_defaults(headless=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {path}")
    return data


def _load_config(args_cli: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    path = Path(args_cli.config).expanduser().resolve()
    return _load_yaml(path), path


def _key(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _dtype_from_name(name: str) -> np.dtype:
    if name != "float32":
        raise ValueError(f"Unsupported command dtype {name!r}; only 'float32' is supported.")
    return np.dtype(np.float32)


def _parse_action_slice(raw_slice: Any, action_dim: int) -> tuple[int, int]:
    if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
        raise ValueError(f"action_slice must be [start, end], got {raw_slice!r}")
    start, end = int(raw_slice[0]), int(raw_slice[1])
    if start < 0 or end <= start or end > action_dim:
        raise ValueError(f"Invalid action_slice [{start}, {end}] for action dim {action_dim}")
    return start, end


def _command_configs(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    commands = config.get("commands", [])
    if not isinstance(commands, list) or not commands:
        raise ValueError("Deploy config must define a non-empty commands list.")
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError(f"Command entries must be mappings, got {type(command)}")
    return commands


def _subscribe_commands(
    *,
    bus: Any,
    namespace: str,
    commands: list[Mapping[str, Any]],
    latest_action: np.ndarray,
    action_lock: threading.Lock,
    logger: logging.Logger,
) -> None:
    action_dim = latest_action.shape[0]

    for command in commands:
        topic = str(command["topic"])
        dtype = _dtype_from_name(str(command.get("dtype", "float32")))
        start, end = _parse_action_slice(command["action_slice"], action_dim)
        command_len = end - start
        key_expr = _key(namespace, topic)

        def _handler(
            key: str,
            payload: bytes,
            *,
            _dtype: np.dtype = dtype,
            _start: int = start,
            _end: int = end,
            _command_len: int = command_len,
        ) -> None:
            del key
            values = np.frombuffer(payload, dtype=_dtype)
            if values.size < _command_len:
                logger.warning(
                    "Ignoring short command on %s: got %d values, need %d",
                    key_expr,
                    values.size,
                    _command_len,
                )
                return
            with action_lock:
                latest_action[_start:_end] = values[:_command_len].astype(np.float32, copy=False)

        bus.subscribe(key_expr, _handler)
        logger.info("Subscribed %s -> action[%d:%d]", key_expr, start, end)


def _publish_observation_value(bus: Any, key: str, value: Any) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to publish observations.") from exc

    if isinstance(value, torch.Tensor):
        bus.publish_torch_tensor(key, value)
        return
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _publish_observation_value(bus, _key(key, str(child_key)), child_value)


def _publish_observations(bus: Any, namespace: str, root: str, observations: Mapping[str, Any]) -> None:
    for group_name, value in observations.items():
        _publish_observation_value(bus, _key(namespace, root, str(group_name)), value)


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("deploy.sim")

    config, config_path = _load_config(args_cli)
    logger.info("Loaded deploy config: %s", config_path)

    logger.info("Launching Isaac Sim app...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    logger.info("Isaac Sim app launched.")

    logger.info("Importing runtime dependencies...")
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from zenoh_bus import ZenohBus

    import isaac_neuromeka.tasks.demo  # noqa: F401
    from isaac_neuromeka.env.vecenv_wrapper import NrmkRlVecEnvWrapper

    logger.info("Runtime dependencies imported.")

    task = str(config["isaaclab_task"])
    communication = config.get("communication", {})
    if not isinstance(communication, Mapping):
        raise ValueError("communication must be a mapping.")
    namespace = str(communication.get("topic_namespace", task))
    zenoh_config = communication.get("zenoh")

    logger.info("Parsing IsaacLab env config for task: %s", task)
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1, use_fabric=True)
    logger.info("Creating Gym environment: %s", task)
    env = gym.make(task, cfg=env_cfg)
    logger.info("Wrapping environment with NrmkRlVecEnvWrapper...")
    env = NrmkRlVecEnvWrapper(env)
    logger.info(
        "Environment ready: num_envs=%d, num_actions=%d, device=%s",
        env.num_envs,
        env.num_actions,
        env.device,
    )

    latest_action = np.zeros(env.num_actions, dtype=np.float32)
    action_lock = threading.Lock()
    observations_cfg = config.get("observations", {})
    obs_root = "obs"
    if isinstance(observations_cfg, Mapping):
        obs_root = str(observations_cfg.get("root", obs_root))

    step_count = 0
    logger.info("Opening Zenoh bus...")
    bus = ZenohBus(zenoh_config if isinstance(zenoh_config, dict) else None)
    logger.info("Zenoh bus ready.")
    _subscribe_commands(
        bus=bus,
        namespace=namespace,
        commands=_command_configs(config),
        latest_action=latest_action,
        action_lock=action_lock,
        logger=logger,
    )

    try:
        logger.info("Starting simulation loop...")
        if not simulation_app.is_running():
            logger.warning("Isaac Sim app is not running before the first simulation step.")
        while simulation_app.is_running():
            with torch.inference_mode():
                start = time.time()
                with action_lock:
                    action_np = latest_action.copy()
                action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32).view(1, -1)
                if env.num_envs != 1:
                    action = action.repeat(env.num_envs, 1)

                _, _, _, infos = env.step(action)
                observations = infos.get("observations", {})
                if isinstance(observations, Mapping):
                    _publish_observations(bus, namespace, obs_root, observations)

                step_count += 1

                if step_count == 1:
                    logger.info("First simulation step complete.")
                if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                    break

                wait_time = env.unwrapped.step_dt - (time.time() - start)
                if args_cli.real_time and wait_time > 0:
                    time.sleep(wait_time)
    finally:
        logger.info("Simulation loop exited after %d steps.", step_count)
        bus.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

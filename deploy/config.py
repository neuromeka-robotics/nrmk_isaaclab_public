from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Deploy config must be a mapping: {config_path}")
    return config, config_path


def key(*parts: object) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def mapping_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping when defined.")
    return value


def command_configs(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    commands = config.get("commands", [])
    if not isinstance(commands, list) or not commands:
        raise ValueError("Deploy config must define a non-empty commands list.")
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError(f"Command entries must be mappings, got {type(command)}")
    return commands


def parse_action_slice(raw_slice: Any, action_dim: int | None = None) -> tuple[int, int]:
    if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
        raise ValueError(f"action_slice must be [start, end], got {raw_slice!r}")

    start, end = int(raw_slice[0]), int(raw_slice[1])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid action_slice [{start}, {end}]")
    if action_dim is not None and end > action_dim:
        raise ValueError(f"Invalid action_slice [{start}, {end}] for action dim {action_dim}")
    return start, end


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

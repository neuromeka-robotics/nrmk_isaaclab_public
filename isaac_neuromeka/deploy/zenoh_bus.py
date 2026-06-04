from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

import zenoh

ENC_RAW = "application/octet-stream"


class ZenohBus:
    """Small Zenoh pub/sub wrapper used by deploy simulations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._log = logging.getLogger("ZenohBus")
        self._lock = threading.Lock()
        self._closed = False
        self._subscribers: list[Any] = []
        self._session = zenoh.open(self._build_config(config))

    def _build_config(self, config: dict[str, Any] | None) -> zenoh.Config:
        if not config:
            return zenoh.Config()
        try:
            return zenoh.Config.from_json5(json.dumps(config))
        except Exception:
            self._log.warning("Failed to build Zenoh config from JSON5; using insert_json5")
            cfg = zenoh.Config()
            for key, value in config.items():
                cfg.insert_json5(key, json.dumps(value))
            return cfg

    def publish(self, key: str, payload: bytes, *, encoding: str | None = None) -> None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"Payload must be bytes-like, got {type(payload)}")
        with self._lock:
            if self._closed:
                raise RuntimeError("ZenohBus is closed.")
            session = self._session
        session.put(key, bytes(payload), encoding=encoding)

    def publish_torch_tensor(self, key: str, tensor: Any, *, encoding: str | None = ENC_RAW) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to publish tensors.") from exc

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        data = tensor.detach().contiguous()
        if data.device.type != "cpu":
            data = data.cpu()
        self.publish(key, data.numpy().tobytes(), encoding=encoding)

    def subscribe(self, key_expr: str, handler: Callable[[str, bytes], None]) -> Any:
        if not callable(handler):
            raise TypeError("Handler must be callable")

        def _safe_listener(sample: Any) -> None:
            try:
                key = str(sample.key_expr)
                payload = sample.payload.to_bytes()
                handler(key, payload)
            except Exception:
                self._log.exception("Exception in subscriber handler for '%s'", key_expr)

        with self._lock:
            if self._closed:
                raise RuntimeError("ZenohBus is closed.")
            subscriber = self._session.declare_subscriber(key_expr, _safe_listener)
            self._subscribers.append(subscriber)
            return subscriber

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            subscribers = list(self._subscribers)
            self._subscribers.clear()

        for subscriber in subscribers:
            if hasattr(subscriber, "undeclare"):
                try:
                    subscriber.undeclare()
                except Exception:
                    self._log.debug("Ignoring subscriber undeclare during close.", exc_info=True)

        if hasattr(session, "close"):
            session.close()

    def __enter__(self) -> ZenohBus:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

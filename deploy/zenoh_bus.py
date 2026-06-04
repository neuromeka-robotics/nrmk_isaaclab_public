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
        self._queryables: list[Any] = []
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

    def subscribe_torch_tensor(
        self,
        key_expr: str,
        handler: Callable[[str, Any], None],
        *,
        dtype: Any | None = None,
    ) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to subscribe to tensors.") from exc

        tensor_dtype = torch.float32 if dtype is None else dtype

        def _tensor_listener(key: str, payload: bytes) -> None:
            try:
                tensor = torch.frombuffer(payload, dtype=tensor_dtype)
                handler(key, tensor)
            except Exception:
                self._log.exception("Exception in tensor subscriber handler for '%s'", key_expr)

        return self.subscribe(key_expr, _tensor_listener)

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

    def query(
        self,
        key: str,
        payload: bytes = b"",
        *,
        timeout_ms: int = 2000,
        encoding: str | None = None,
        target: Any | None = None,
    ) -> tuple[bool, bytes | None, str | None]:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"Payload must be bytes-like, got {type(payload)}")

        with self._lock:
            if self._closed:
                raise RuntimeError("ZenohBus is closed.")
            session = self._session

        replies = session.get(
            key,
            payload=bytes(payload),
            encoding=encoding,
            timeout=timeout_ms / 1000.0,
            target=target,
        )

        for reply in replies:
            if reply.ok is not None:
                return True, reply.ok.payload.to_bytes(), None

            if reply.err is not None:
                try:
                    error_message = reply.err.payload.to_string()
                except Exception:
                    error_message = f"Unknown error payload: {reply.err.payload}"
                return False, None, error_message

        return False, None, "timeout_or_no_replies"

    def declare_queryable(
        self,
        key: str,
        handler: Callable[[bytes], bytes],
        *,
        complete: bool = False,
    ) -> Any:
        if not callable(handler):
            raise TypeError("Handler must be callable")

        def _on_query(query: Any) -> None:
            try:
                request = query.payload.to_bytes() if query.payload is not None else b""
                response = handler(request)
                if not isinstance(response, (bytes, bytearray, memoryview)):
                    self._log.error("Queryable handler for '%s' returned non-bytes: %s", key, type(response))
                    query.reply_err("Internal Server Error: Invalid return type")
                    return
                query.reply(query.key_expr, bytes(response))
            except Exception as exc:
                self._log.exception("Queryable handler failed for '%s'", key)
                query.reply_err(str(exc))

        with self._lock:
            if self._closed:
                raise RuntimeError("ZenohBus is closed.")
            queryable = self._session.declare_queryable(key, _on_query, complete=complete)
            self._queryables.append(queryable)
            return queryable

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            subscribers = list(self._subscribers)
            queryables = list(self._queryables)
            self._subscribers.clear()
            self._queryables.clear()

        for subscriber in subscribers:
            if hasattr(subscriber, "undeclare"):
                try:
                    subscriber.undeclare()
                except Exception:
                    self._log.debug("Ignoring subscriber undeclare during close.", exc_info=True)

        for queryable in queryables:
            if hasattr(queryable, "undeclare"):
                try:
                    queryable.undeclare()
                except Exception:
                    self._log.debug("Ignoring queryable undeclare during close.", exc_info=True)

        if hasattr(session, "close"):
            session.close()

    def __enter__(self) -> ZenohBus:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, List, Optional, Tuple, Union  # noqa: F401

import zenoh


class ZenohBus:
    """
    Opinionated Zenoh wrapper for robust RPC and Pub/Sub.

    Refinements:
      - Handler signature: (key: str, payload: bytes) -> None
      - Thread-safe lifecycle management without blocking the hot path.
      - Enforces bytes-only payloads for strict serialization control.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._log = logging.getLogger("ZenohBus")
        self._lock = threading.Lock()  # Protects _subs and _queryables lists only
        self._subs: List[zenoh.Subscriber] = []
        self._queryables: List[zenoh.Queryable] = []

        # Initialize session
        self._session = zenoh.open(self._build_config(config))

    def _build_config(self, config: Optional[dict]) -> zenoh.Config:
        if not config:
            return zenoh.Config()
        try:
            return zenoh.Config.from_json5(json.dumps(config))
        except Exception:
            self._log.warning("Failed to dump config to JSON5, falling back to manual insert")
            cfg = zenoh.Config()
            for k, v in config.items():
                cfg.insert_json5(k, str(v))
            return cfg

    # -------------------------
    # Pub/Sub
    # -------------------------
    def publish(self, key: str, payload: bytes, *, encoding: Optional[str] = None) -> None:
        """
        Fire-and-forget publish. No locking on the hot path.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"Payload must be bytes-like, got {type(payload)}")

        # Zenoh put is thread-safe internally
        self._session.put(key, bytes(payload), encoding=encoding)

    def subscribe(self, key_expr: str, handler: Callable[[str, bytes], None]) -> zenoh.Subscriber:
        """
        Subscribe to a key expression.

        Args:
            key_expr: The Zenoh key expression (supports wildcards, e.g., 'robot/**')
            handler: Callback function with signature (key: str, payload: bytes) -> None
        """
        if not callable(handler):
            raise TypeError("Handler must be callable")

        def _safe_listener(sample: zenoh.Sample) -> None:
            # We catch everything here to prevent the Zenoh rust-thread from panicking
            try:
                # Extract key string (useful for wildcards)
                key = str(sample.key_expr)
                payload = sample.payload.to_bytes()
                handler(key, payload)
            except Exception:
                self._log.exception(f"Exception in subscriber handler for '{key_expr}'")

        sub = self._session.declare_subscriber(key_expr, _safe_listener)

        with self._lock:
            self._subs.append(sub)
        return sub

    # -------------------------
    # Query / Queryable (RPC Pattern)
    # -------------------------
    def query(
        self,
        key: str,
        payload: bytes = b"",
        *,
        timeout_ms: int = 2000,
        encoding: Optional[str] = None,
        target: Optional[zenoh.QueryTarget] = None,
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Performs a request-response (RPC) call.

        Behavior:
          - Returns the FIRST successful reply received.
          - Ignores subsequent replies (if 1-to-N).

        Returns:
          (success, payload_bytes, error_message)
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("Payload must be bytes-like")

        # Zenoh get returns a stream (iterator) of replies
        replies = self._session.get(
            key,
            payload=bytes(payload),
            encoding=encoding,
            timeout=timeout_ms / 1000.0,
            target=target,
        )

        for reply in replies:
            # Case 1: The replier processed it successfully
            if reply.ok is not None:
                return True, reply.ok.payload.to_bytes(), None

            # Case 2: The replier sent an error explicitly
            if reply.err is not None:
                try:
                    err_msg = reply.err.payload.to_string()
                except Exception:
                    err_msg = f"Unknown error payload: {reply.err.payload}"
                return False, None, err_msg

        # Case 3: No replies received within timeout
        return False, None, "timeout_or_no_replies"

    def declare_queryable(
        self,
        key: str,
        handler: Callable[[bytes], bytes],
        *,
        complete: bool = False,
    ) -> zenoh.Queryable:
        """
        Registers a function to answer queries on `key`.

        Args:
            handler: (request_bytes) -> response_bytes
            complete: If True, this queryable claims 'completeness' for the key expression.
        """
        if not callable(handler):
            raise TypeError("Handler must be callable")

        def _on_query(query: zenoh.Query) -> None:
            try:
                req = query.payload.to_bytes() if query.payload is not None else b""
                resp = handler(req)

                if not isinstance(resp, (bytes, bytearray, memoryview)):
                    self._log.error(f"Queryable handler for '{key}' returned non-bytes: {type(resp)}")
                    query.reply_err("Internal Server Error: Invalid return type")
                    return

                query.reply(query.key_expr, bytes(resp))
            except Exception as e:
                self._log.exception(f"Queryable handler failed for '{key}'")
                query.reply_err(str(e))

        q = self._session.declare_queryable(key, _on_query, complete=complete)

        with self._lock:
            self._queryables.append(q)
        return q

    # -------------------------
    # Lifecycle
    # -------------------------
    def close(self) -> None:
        """
        Cleanly undeclare all subscribers/queryables and close the session.
        """
        self._log.info("Closing ZenohBus...")
        with self._lock:
            subs = self._subs[:]
            qs = self._queryables[:]
            self._subs.clear()
            self._queryables.clear()

        # Undeclare strictly
        for s in subs:
            try:
                s.undeclare()
            except Exception as e:
                self._log.warning(f"Error undeclaring subscriber: {e}")

        for q in qs:
            try:
                q.undeclare()
            except Exception as e:
                self._log.warning(f"Error undeclaring queryable: {e}")

        try:
            self._session.close()
        except Exception as e:
            self._log.warning(f"Error closing session: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# TODO: shm integration
class ShmPublisher:
    def __init__(self, session: zenoh.Session, pool_bytes: int = 256 * 1024 * 1024) -> None:
        self._session = session
        self._provider = zenoh.shm.ShmProvider.default_backend(pool_bytes)

    def put(self, key: str, payload: bytes, *, encoding=None) -> None:
        # Try SHM; if it fails, fall back to normal bytes publish.
        try:
            buf = self._provider.alloc(
                len(payload),
                policy=zenoh.shm.BlockOn(zenoh.shm.GarbageCollect()),
            )
            buf[:] = payload
            self._session.put(key, buf, encoding=encoding)
        except Exception:
            self._session.put(key, payload, encoding=encoding)


# def cb(sample: zenoh.Sample) -> None:
#     payload = sample.payload
#     if payload.as_shm() is not None:
#         pass  # came via SHM
#     data = payload.to_bytes()

# import cv2
import numpy as np
import torch
from communication.zenoh_bus import ZenohBus


def _to_uint8_image(t: torch.Tensor) -> torch.Tensor:
    # Expect float32 in [0,1] or [0,255] (we handle both).
    t = t.detach().contiguous()
    if t.dtype != torch.float32:
        raise TypeError(f"expected float32 image tensor, got {t.dtype}")

    # Clamp and scale if it looks like normalized [0,1]
    maxv = float(t.max().item()) if t.numel() > 0 else 0.0
    if maxv <= 1.5:
        t = (t.clamp(0.0, 1.0) * 255.0).round()
    else:
        t = t.clamp(0.0, 255.0).round()

    return t.to(torch.uint8)


def _depth_to_uint16_mm(depth: torch.Tensor) -> torch.Tensor:
    # float32 meters -> uint16 millimeters (0..65535mm = 65.535m)
    depth = depth.detach().contiguous()
    if depth.dtype != torch.float32:
        raise TypeError(f"expected float32 depth tensor, got {depth.dtype}")

    mm = (depth * 1000.0).round()
    mm = mm.clamp(0.0, 65535.0)
    return mm.to(torch.uint16)


ENC_RAW = "application/octet-stream"
ENC_META = "text/plain"


def _meta_of(arr: np.ndarray) -> str:
    # Example: dtype=float32;shape=1,480,640,3
    return f"dtype={arr.dtype.name};shape={','.join(map(str, arr.shape))}"


def publish_array(
    bus: ZenohBus,
    key: str,
    t: torch.Tensor,
    *,
    publish_meta: bool = False,
) -> None:
    arr = t.detach().contiguous()
    if arr.device.type != "cpu":
        arr = arr.cpu()
    arr = arr.numpy()

    bus.publish(key, arr.tobytes(), encoding=ENC_RAW)

    if publish_meta:
        bus.publish(key + "/meta", _meta_of(arr).encode("utf-8"), encoding=ENC_META)


def publish_array_shm(
    bus: ZenohBus,
    key: str,
    t: torch.Tensor,
    *,
    shm_provider,
    publish_meta: bool = False,
    encoding: str = ENC_RAW,
) -> None:
    """
    Publish a tensor using Zenoh SHM.

    Requirements:
      - eclipse-zenoh (zenoh-python) >= 1.6.x
      - shm_provider: zenoh.shm.ShmProvider instance, created by caller

    Notes:
      - This still copies once: tensor -> SHM buffer.
      - Transport can be zero-copy for same-host subscribers that support SHM.
    """
    arr = t.detach().contiguous()
    if arr.device.type != "cpu":
        arr = arr.cpu()
    arr = arr.numpy()
    data = arr.tobytes()

    # Allocate shared-memory buffer and copy once into it
    buf = shm_provider.alloc(
        len(data),
        policy=(
            shm_provider.__class__.BlockOn(shm_provider.__class__.GarbageCollect())
            if hasattr(shm_provider.__class__, "BlockOn")
            else None
        ),
    )
    buf[:] = data

    bus.publish(key, buf, encoding=encoding)

    if publish_meta:
        bus.publish(key + "/meta", _meta_of(arr).encode("utf-8"), encoding=ENC_META)

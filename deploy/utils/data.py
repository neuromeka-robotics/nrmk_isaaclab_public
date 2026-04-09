import numpy as np
import torch


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


import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from threading import Lock

class SensorVisualizer:
    """Stores the latest sensor frames and renders them in Matplotlib and Open3D windows."""

    def __init__(
        self,
        width: int,
        height: int,
        intrinsics: dict[str, float],
        clipping_range: tuple[float, float] = (0.1, 10.0),
        pcl_subsample_stride: int = 1
    ) -> None:
        # Set params
        self.width = width
        self.height = height
        self.intrinsics = intrinsics
        assert all(key in self.intrinsics.keys() for key in ["fx", "fy", "cx", "cy"]), "Camera intrinsics not all present."
        self.clipping_range = clipping_range
        self.pointcloud_stride = max(1, pcl_subsample_stride)
        
        self._lock = Lock()
        self._image: np.ndarray | None = None
        self._depth_image: np.ndarray | None = None
        self._has_new_data = False
        
        # Set image viewer
        plt.ion()
        self._figure = plt.figure(figsize=(8, 4))
        plt.show(block=False)
        
        # Set point cloud viewer
        self._point_cloud = o3d.geometry.PointCloud()
        self._o3d_vis = o3d.visualization.Visualizer()
        self._o3d_vis.create_window(window_name="Point Cloud", width=640 * 2, height=360 * 2)
        self._coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self._coord_frame.rotate(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64), center=np.array([0, 0, 0], dtype=np.float64))  # match camera coordinates
        self._o3d_vis.add_geometry(self._point_cloud)
        self._o3d_vis.add_geometry(self._coord_frame)
        view_control = self._o3d_vis.get_view_control()
        view_control.set_constant_z_far(1000)
        
    def close(self) -> None:
        self._o3d_vis.destroy_window()

    def update_image(self, payload: bytes) -> None:
        arr = np.frombuffer(payload, dtype=np.float32)
        image = self._reshape_image(arr)
        if image is None:
            return
        image_uint8 = _to_uint8_image(torch.from_numpy(image.copy())).cpu().numpy()
        with self._lock:
            self._image = image_uint8
            self._has_new_data = True

    def update_depth(self, payload: bytes) -> None:
        arr = np.frombuffer(payload, dtype=np.float32)
        depth = self._reshape_depth(arr)
        if depth is None:
            return
        with self._lock:
            self._depth_image = depth.copy()
            self._has_new_data = True

    def draw(self) -> None:
        # Retrieve the latest sensor data
        with self._lock:
            image = None if self._image is None else self._image.copy()
            depth_image = None if self._depth_image is None else self._depth_image.copy()
            has_new_data = self._has_new_data
            self._has_new_data = False

        if not has_new_data or (image is None and depth_image is None):
            self._o3d_vis.poll_events()
            self._o3d_vis.update_renderer()
            return

        points, colors = self._build_point_cloud(image, depth_image)

        # Visualize
        self._figure.clf()
        ax_rgb = self._figure.add_subplot(1, 2, 1)
        if image is not None:
            ax_rgb.imshow(image)
            ax_rgb.set_title("RGB Image")
        ax_rgb.axis("off")

        ax_depth = self._figure.add_subplot(1, 2, 2)
        if depth_image is not None:
            ax_depth.imshow(depth_image, cmap="gray")
            ax_depth.set_title("Depth Image")
        ax_depth.axis("off")

        self._figure.tight_layout()
        self._figure.canvas.draw_idle()

        if points is not None:
            self._point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if colors is not None:
                self._point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            
            # transform for open3d visualization
            R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
            self._point_cloud.rotate(R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
            self._o3d_vis.update_geometry(self._point_cloud)

        self._o3d_vis.poll_events()
        self._o3d_vis.update_renderer()
        plt.pause(0.001)
        
    def _reshape_image(self, arr: np.ndarray) -> np.ndarray | None:
        pixel_count = self.height * self.width
        if arr.size == pixel_count * 3:
            return arr.reshape(self.height, self.width, 3)
        return None

    def _reshape_depth(self, arr: np.ndarray) -> np.ndarray | None:
        pixel_count = self.height * self.width
        if arr.size == pixel_count:
            return arr.reshape(self.height, self.width)
        return None
    
    def _build_point_cloud(
        self, image: np.ndarray | None, depth_image: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None] | tuple[None, None]:
        if depth_image is None:
            return None, None

        stride = self.pointcloud_stride
        sampled_depth = depth_image[::stride, ::stride]
        valid = np.isfinite(sampled_depth)
        valid &= sampled_depth >= self.clipping_range[0]
        valid &= sampled_depth <= self.clipping_range[1]
        if not np.any(valid):
            return None, None

        grid_y, grid_x = np.mgrid[0 : self.height : stride, 0 : self.width : stride]

        z = sampled_depth[valid]
        x = ((grid_x[valid] - self.intrinsics["cx"]) / self.intrinsics["fx"]) * z
        y = ((grid_y[valid] - self.intrinsics["cy"]) / self.intrinsics["fy"]) * z
        points = np.column_stack((x, y, z))

        colors = None
        if image is not None:
            sampled_image = image[::stride, ::stride]
            colors = sampled_image[valid].astype(np.float32) / 255.0

        return points, colors

# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import builtins

# python
import os
from typing import TYPE_CHECKING



# isaac-lab
import isaaclab.sim as sim_utils

from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

# WARP
import warp as wp
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp import raycast_mesh

## 
import trimesh
from isaaclab.utils import configclass
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

@configclass
class MeshTerrainImporterCfg(TerrainImporterCfg):
    obj_dir: str = ""
    node_density: float = 1.0 # number of nodes per m^2
    terrain_type= "usd"


class MeshTerrainImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: MeshTerrainImporterCfg

    mesh: trimesh.Trimesh

    def __init__(self, cfg: MeshTerrainImporterCfg) -> None:

        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        collision_approximation = "none" 
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)


        usd_path = "/home/nrmk/Documents/usd_test/terrain.usd"
            # Create Mesh converter config
        mesh_converter_cfg = MeshConverterCfg(
            mass_props=None,
            rigid_props=None,
            collision_props=collision_props,
            asset_path="/home/nrmk/Documents/ETH_LEE_H_with_terrace_cropped/box_000_000_000_00000.obj",
            force_usd_conversion=True,
            usd_dir=os.path.dirname(usd_path),
            usd_file_name=os.path.basename(usd_path),
            make_instanceable=True,
            collision_approximation=collision_approximation,
        )
        
        self.mesh_converter = MeshConverter(cfg=mesh_converter_cfg)



        self.load_terrain_mesh(cfg.obj_dir)

        self.import_usd("terrain", usd_path)
        # self.import_mesh(name="terrain", mesh = self.mesh)

        # assign randomly
        num_envs = self.cfg.num_envs
        idx = torch.randint(
            low=0, high=self.terrain_origins.shape[0], size=(num_envs,), device=self.device
        )
        self.env_origins = self.terrain_origins[idx]

        self.set_debug_vis(True)



    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization of the terrain importer.

        Args:
            debug_vis: Whether to visualize the terrain origins.

        Returns:
            Whether the debug visualization was successfully set. False if the terrain
            importer does not support debug visualization.

        Raises:
            RuntimeError: If terrain origins are not configured.
        """
        # create a marker if necessary

        FRAME_MARKER_CFG = VisualizationMarkersCfg(
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.2, 0.2, 0.2),
                )
            }
        )
        
        if debug_vis:
            if not hasattr(self, "origin_visualizer"):
                self.origin_visualizer = VisualizationMarkers(
                    cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/TerrainOrigin")
                )
                if self.terrain_origins is not None:
                    self.origin_visualizer.visualize(self.terrain_origins.reshape(-1, 3))
                elif self.env_origins is not None:
                    self.origin_visualizer.visualize(self.env_origins.reshape(-1, 3))
                else:
                    raise RuntimeError("Terrain origins are not configured.")
            # set visibility
            self.origin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.set_visibility(False)
        # report success
        return True
    
    def load_terrain_mesh(self, obj_dir: str):
        # paths = [os.path.join(obj_dir, file) for file in os.listdir(obj_dir) if file.endswith(".obj")]
        # meshes = [trimesh.load(path, force="mesh", skip_materials=True) for path in paths]
        # self.mesh = trimesh.util.concatenate(meshes)

        self.mesh = trimesh.load("/home/nrmk/Documents/ETH_LEE_H_with_terrace_cropped/box_000_000_000_00000.obj", force="mesh")

        self.mesh.visual.vertex_colors = None

        self.terrain_origins = self.sample_nodes_from_mesh()

        


    def sample_nodes_from_mesh(self):
        """
        Sample nodes from the mesh.
        """

        sample_count = int(self.mesh.area * self.cfg.node_density)
        face_weight = self.mesh.area_faces
        samples, face_index = trimesh.sample.sample_surface(self.mesh, sample_count, face_weight)


        face_ids = torch.tensor(face_index, device=self.device, dtype=torch.int64)
        positions = torch.tensor(samples, device=self.device, dtype=torch.float32)
        normals = torch.tensor(self.mesh.face_normals[face_index, :], device=self.device)

        # check gravity alignment
        gravity = torch.tensor([0.0, 0.0, -1.0]).to(normals)
        mask = torch.abs(torch.arccos(torch.matmul(normals, -gravity))) < 0.785 # # 45 degrees 

        face_ids = face_ids[mask]
        positions = positions[mask, :]
        normals = normals[mask, :]

        ## Find flat patches
        mesh_to_wp = convert_to_warp_mesh(self.mesh.vertices, self.mesh.faces, device="cuda")


        positions, face_ids = self.filter_flat(
            mesh_to_wp,
            positions,
            face_ids,
            [0.5, 0.75],
        )


        return positions
    



    def filter_flat(self,
        wp_mesh: wp.Mesh,
        sampled_points: torch.Tensor,
        face_ids: torch.Tensor,
        patch_radius: float | list[float],
    ) -> torch.Tensor:

        # set device to warp mesh device
        device = wp.device_to_torch(wp_mesh.device)

        # resolve inputs to consistent type
        # -- patch radii
        if isinstance(patch_radius, float):
            patch_radius = [patch_radius]


        # create a circle of points around (0, 0) to query validity of the patches
        # the ring of points is uniformly distributed around the circle
        angle = torch.linspace(0, 2 * np.pi, 10, device=device)
        query_x = []
        query_y = []
        for radius in patch_radius:
            query_x.append(radius * torch.cos(angle))
            query_y.append(radius * torch.sin(angle))
        query_x = torch.cat(query_x).unsqueeze(1)  # dim: (num_radii * 10, 1)
        query_y = torch.cat(query_y).unsqueeze(1)  # dim: (num_radii * 10, 1)

        # dim: (num_radii * 10, 3)
        query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

        num_points = sampled_points.shape[0]
        # create buffers
        # -- a buffer to store indices of points that are not valid
        points_ids = torch.arange(num_points, device=device)
        

        # dim: (num_points, num_radii * 10, 3)
        scan_points = sampled_points.unsqueeze(1) + query_points
        scan_points[..., 2] += 1.0 # Raycast from 1.0 above the patch to find the height

        # ray-cast direction is downwards
        scan_dirs = torch.zeros_like(scan_points)
        scan_dirs[..., 2] = -1.0


        ray_hits = raycast_mesh(scan_points.view(-1, 3), scan_dirs.view(-1, 3), wp_mesh)[0]

        # resphape_back
        ray_hits_per_point = ray_hits.reshape(scan_points.shape[0], -1, 3)  # dim: (num_points, num_radii * 10, 3)
        heights = ray_hits_per_point[..., 2]  # dim: (num_points, num_radii * 10)

        heights_from_sampled_points = sampled_points[:, 2].unsqueeze(1) - heights  # dim: (num_points, num_radii * 10)

        print(heights_from_sampled_points)
        valid = torch.all(
            torch.logical_and(
                heights_from_sampled_points >= -0.1,
                heights_from_sampled_points <= 0.1
            ),
            dim=1
        )  # dim: (num_points,)
        
        points_valid = sampled_points[valid, :]  # dim: (num_valid_points, 3)
        face_ids = face_ids[valid]  # dim: (num_valid_points,)

        return points_valid, face_ids
import trimesh
from isaaclab.utils import configclass
import torch
import numpy as np

# from isaaclab.terrains.utils import find_flat_patches
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp import raycast_mesh

import warp as wp
wp.init()

# Replace with the path to your OBJ file
obj_path = "/home/nrmk/Documents/ETH_LEE_H_with_terrace_cropped/box_000_000_000_00000.obj"

# Load the OBJ file
mesh: trimesh.Trimesh = trimesh.load(obj_path, force='mesh')

# TODO: merge and clean the mesh if necessary

# Add a small sphere at the origin to visualize it
origin_sphere = trimesh.creation.icosphere(radius=0.5)
origin_sphere.apply_translation([0, 0, 0])
# origin_sphere.apply_translation([ 21.50417656, -40.84542959 , -0.64960542])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

count = int(0.5 * mesh.area) # num nodes / m^2

face_weight = mesh.area_faces
samples, face_index = trimesh.sample.sample_surface(mesh, count, face_weight)

face_ids = torch.tensor(face_index, device=device, dtype=torch.int64)
positions = torch.tensor(samples, device=device, dtype=torch.float32)
normals = torch.tensor(mesh.face_normals[face_index, :], device=device)



# check gravity alignment
gravity = torch.tensor([0.0, 0.0, -1.0]).to(normals)
mask = torch.abs(torch.arccos(torch.matmul(normals, -gravity))) < 0.75 


face_ids = face_ids[mask]
positions = positions[mask, :]
normals = normals[mask, :]

## Find flat patches
wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device="cuda")



def filter_flat(
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

    return points_valid, face_ids, scan_points


positions, face_ids, scan_points = filter_flat(
    wp_mesh=wp_mesh,
    sampled_points= positions,
    face_ids=face_ids,
    patch_radius=[0.5, 0.75],
)



# visualize positions in trimesh
position_spheres = []
for i in range(positions.shape[0]):
	sphere = trimesh.creation.icosphere(radius=0.2)
	pos = positions[i, :].cpu().numpy()
	sphere.apply_translation(pos)
	sphere.visual.face_colors = [255, 0, 0, 100]  # Red color with some transparency
	position_spheres.append(sphere)
	
# # visualize positions in trimesh
# position_spheres2= []
# debug_points = scan_points.reshape(-1, 3)  # dim: (num_points * num_radii * 10, 3)
# for i in range(debug_points.shape[0]):
#     sphere = trimesh.creation.icosphere(radius=0.1)
#     pos = debug_points[i, :].cpu().numpy()
#     sphere.apply_translation(pos)
#     sphere.visual.face_colors = [0, 0, 255, 255]  # Red color with some transparency
#     position_spheres2.append(sphere)
	



# scene = trimesh.Scene([mesh, origin_sphere] + position_spheres + position_spheres2)
scene = trimesh.Scene([mesh, origin_sphere] + position_spheres)
scene.show()


    # def import_mesh(self, name: str, mesh: trimesh.Trimesh):
    #     """Import a mesh into the simulator.

    #     The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
    #     contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

    #     Args:
    #         name: The name of the imported terrain. This name is used to create the USD prim
    #             corresponding to the terrain.
    #         mesh: The mesh to import.

    #     Raises:
    #         ValueError: If a terrain with the same name already exists.
    #     """
    #     # create prim path for the terrain
    #     prim_path = self.cfg.prim_path + f"/{name}"
    #     # check if key exists
    #     if prim_path in self.terrain_prim_paths:
    #         raise ValueError(
    #             f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
    #         )
    #     # store the mesh name
    #     self.terrain_prim_paths.append(prim_path)

    #     # import the mesh
    #     create_prim_from_mesh(
    #         prim_path, mesh, visual_material=self.cfg.visual_material, physics_material=self.cfg.physics_material
    #     )


    # paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".obj")]
    # meshes = [trimesh.load(path, force="mesh", skip_materials=True) for path in paths]
    # mesh = trimesh.util.concatenate(meshes)
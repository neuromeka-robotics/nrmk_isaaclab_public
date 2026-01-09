from isaac_neuromeka.env.vecenv_wrapper import NrmkRlVecEnvWrapper
import torch

class EnvWrapper:
    def __init__(self, env, debug_vis=False):
        self.env = NrmkRlVecEnvWrapper(env)
        self.unwrapped = self.env.unwrapped
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.control_dt = self.unwrapped.step_dt
        self.num_actions = self.env.num_actions
        self.obs_dict = {}
        self.env.reset()
        self.debug_vis = debug_vis

        self._init_debug_vis()

    @property
    def env_instance(self):
        return self.env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def set_joint_state(self, joint_position, joint_velocity):
        self.unwrapped.robot.write_joint_state_to_sim(
            position=joint_position, velocity=joint_velocity
        )

    def update_command(self, command):
        cmd_term = self.env.unwrapped.command_manager.get_term("ee_pose")
        t_device = cmd_term.pose_command_b.device
        t_dtype = cmd_term.pose_command_b.dtype
        cmd_term.pose_command_b = torch.from_numpy(command[np.newaxis, :]).to(
            device=t_device, dtype=t_dtype
        )
        # self.env.unwrapped.command_manager.get_term("contact_mode").contact_mode[0] = float(self.contact_command)

    def _init_debug_vis(self):
        if self.debug_vis:
            from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
            import isaaclab.sim as sim_utils

            # Pointcloud visualizer
            pc_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                prim_path="/Visuals/RayCaster",
                markers={
                    "hit": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0)
                        ),
                    ),
                },
            )
            self.point_visualizer = VisualizationMarkers(pc_visualizer_cfg)

            # Voxels visualizer
            voxel_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Voxels",
                markers={
                    "voxel": sim_utils.CuboidCfg(
                        size=(0.05, 0.05, 0.05),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.0, 1.0)
                        ),
                    ),
                },
            )
            self.voxel_visualizer = VisualizationMarkers(voxel_visualizer_cfg)
            
            # Pointcloud visualizer
            pc_visualizer_cfg2: VisualizationMarkersCfg = VisualizationMarkersCfg(
                prim_path="/Visuals/RayCaster",
                markers={
                    "hit": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0)
                        ),
                    ),
                },
            )
            self.point_visualizer2 = VisualizationMarkers(pc_visualizer_cfg2)
            
    def update_debug_vis(self, **kwargs):
        if self.debug_vis:
            self.point_visualizer.visualize(kwargs.get("point_cloud"))
            self.voxel_visualizer.visualize(kwargs.get("voxels"))
            self.point_visualizer2.visualize(kwargs.get("point_cloud2"))

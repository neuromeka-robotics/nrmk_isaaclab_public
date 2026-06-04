import numpy as np
from pynput.keyboard import Key
from rl_controller.utils.keyboard_utils import KeyboardListener
from rl_controller.utils.math_helpers import MathFunc


class KeyboardPoseCommand:
    class DeltaCommand:
        pos_x: float
        pos_y: float
        pos_z: float
        yaw: float

    def __init__(self, config, **kwargs):
        command_range, default_ee_pose = self._load_config(config)

        self.command_range = command_range
        self.default_ee_pose = default_ee_pose
        self.make_quat_unique = False

        self.keyboard_listener = KeyboardListener(
            key_targets=[Key.left, Key.up, Key.right, Key.down, "a", "w", "d", "s", "c"]
        )

        n_bins = 15
        self.delta_command = self.DeltaCommand()
        self.delta_command.pos_x = (command_range["x"][1] - command_range["x"][0]) / n_bins
        self.delta_command.pos_y = (command_range["y"][1] - command_range["y"][0]) / n_bins
        self.delta_command.pos_z = (command_range["z"][1] - command_range["z"][0]) / n_bins
        self.delta_command.yaw = (command_range["yaw"][1] - command_range["yaw"][0]) / n_bins

        self.pose_command_b = np.zeros(7)  # position + orientation (i.e., quaternion)
        self.pose_command_b[3] = 1.0
        self.euler_angles = np.zeros(3)
        self.contact_command = False

        self.reset_command()

    def reset_command(self):
        # initialize to default command
        # -- position
        self.pose_command_b[0] = self.default_ee_pose[0]
        self.pose_command_b[1] = self.default_ee_pose[1]
        self.pose_command_b[2] = self.default_ee_pose[2]
        # -- orientation
        if len(self.default_ee_pose) == 6:
            self.euler_angles[0] = self.default_ee_pose[3]
            self.euler_angles[1] = self.default_ee_pose[4]
            self.euler_angles[2] = self.default_ee_pose[5]
        quat = MathFunc.euler_xyz_to_quat(self.euler_angles[0], self.euler_angles[1], self.euler_angles[2])
        self.pose_command_b[3:] = MathFunc.quat_unique(quat) if self.make_quat_unique else quat

    def update(self):
        keyboard_data = self.keyboard_listener.get_key_states()
        if keyboard_data["updated"]:
            # update position
            if keyboard_data["value"][Key.left]:
                self.pose_command_b[0] -= self.delta_command.pos_x
            if keyboard_data["value"][Key.right]:
                self.pose_command_b[0] += self.delta_command.pos_x
            if keyboard_data["value"][Key.up]:
                self.pose_command_b[1] -= self.delta_command.pos_y
            if keyboard_data["value"][Key.down]:
                self.pose_command_b[1] += self.delta_command.pos_y

            if keyboard_data["value"]["s"]:
                self.pose_command_b[2] -= self.delta_command.pos_z
            if keyboard_data["value"]["w"]:
                self.pose_command_b[2] += self.delta_command.pos_z

            self.pose_command_b[:3] = np.clip(
                self.pose_command_b[:3],
                a_min=np.array([self.command_range["x"][0], self.command_range["y"][0], self.command_range["z"][0]]),
                a_max=np.array([self.command_range["x"][1], self.command_range["y"][1], self.command_range["z"][1]]),
            )

            # update orientation
            if keyboard_data["value"]["a"]:
                self.euler_angles[2] -= self.delta_command.yaw
            if keyboard_data["value"]["d"]:
                self.euler_angles[2] += self.delta_command.yaw

            self.euler_angles[2] = np.clip(
                self.euler_angles[2], a_min=self.command_range["yaw"][0], a_max=self.command_range["yaw"][1]
            )

            quat = MathFunc.euler_xyz_to_quat(self.euler_angles[0], self.euler_angles[1], self.euler_angles[2])
            self.pose_command_b[3:] = MathFunc.quat_unique(quat) if self.make_quat_unique else quat

            # toggle contact
            if keyboard_data["value"]["c"]:
                self.contact_command = not self.contact_command

    def _load_config(self, config_dict):
        command_range = {}
        command_range["x"] = np.array(config_dict["command_range"]["x"])
        command_range["y"] = np.array(config_dict["command_range"]["y"])
        command_range["z"] = np.array(config_dict["command_range"]["z"])

        default_ee_pose = np.array(config_dict["default_ee_pose"])

        if len(default_ee_pose) == 6:
            command_range["roll"] = np.array(config_dict["command_range"]["roll"]) + default_ee_pose[3]
            command_range["pitch"] = np.array(config_dict["command_range"]["pitch"]) + default_ee_pose[4]
            command_range["yaw"] = np.array(config_dict["command_range"]["yaw"]) + default_ee_pose[5]
        else:
            command_range["roll"] = np.array(config_dict["command_range"]["roll"])
            command_range["pitch"] = np.array(config_dict["command_range"]["pitch"])
            command_range["yaw"] = np.array(config_dict["command_range"]["yaw"])

        return command_range, default_ee_pose

    @property
    def command(self):
        return self.pose_command_b

    @property
    def contact_mode(self):
        return self.contact_command

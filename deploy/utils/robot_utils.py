import time

import numpy as np
from neuromeka import control_msgs
from sensor_drivers.utils.math_helpers import MathFunc


# OpState:
# SYSTEM_OFF(0), SYSTEM_ON(1), VIOLATE(2), RECOVER_HARD(3), RECOVER_SOFT(4),
# IDLE(5), MOVING(6), TEACHING(7), COLLISION(8), STOP_AND_OFF(9),
# COMPLIANCE(10), BRAKE_CONTROL(11), SYSTEM_RESET(12), SYSTEM_SWITCH(13),
# VIOLATE_HARD(15), MANUAL_RECOVER(16), TELE_OP(17)
class ROBOT_STATE:
    IDLE = 5
    MOVE = 6
    DIRECT_TEACHING = 7
    TELE_OP = 17
    STOP = 0
    VIOLATE = 2
    VIOLATE_HARD = 15
    COLLISION = 8
    RECOVER_HARD = 3
    RECOVER_SOFT = 4
    MANUAL_RECOVER = 16

    @staticmethod
    def in_failure_state(robot_state):
        return robot_state in [
            ROBOT_STATE.STOP,
            ROBOT_STATE.VIOLATE,
            ROBOT_STATE.VIOLATE_HARD,
            ROBOT_STATE.COLLISION,
        ]

    @staticmethod
    def in_recovery_state(robot_state):
        return robot_state in [
            ROBOT_STATE.RECOVER_HARD,
            ROBOT_STATE.RECOVER_SOFT,
            ROBOT_STATE.MANUAL_RECOVER,
        ]


def set_recovery(robot_client):
    while True:
        op_state = robot_client.get_robot_data()["op_state"]
        if ROBOT_STATE.in_failure_state(robot_state=op_state):
            print("Robot failure state. Recover...")
            robot_client.recover()
            time.sleep(1.0)
        elif ROBOT_STATE.in_recovery_state(robot_state=op_state):
            print("Robot recovery state. Wait...")
            time.sleep(1.0)
        else:
            break


def set_teleop(robot_client):
    time_limit = 4.0
    start_time = time.time()
    while time.time() - start_time < time_limit:
        robot_state = robot_client.get_robot_data()["op_state"]
        print("ROBOT STATE", robot_state)
        if robot_state == ROBOT_STATE.TELE_OP:
            break
        robot_client.stop_teleop()
        robot_client.start_teleop(method=control_msgs.TELE_JOINT_ABSOLUTE)
        print("Starting teleop...")
        time.sleep(0.25)


def compute_joint_target_with_ik(
    robot_client,
    target_ee_pos,
    collision_state,
    rl_joint_target_rad,
    ee_error_threshold=0.3,
):
    ee_pos = np.array(robot_client.get_robot_data()["p"])
    joint_pos_deg = MathFunc.rad_to_degree(rl_joint_target_rad).flatten()

    ee_pos_m = MathFunc.mm_to_m(ee_pos[:3])
    pos_error = np.linalg.norm(target_ee_pos[:3] - ee_pos_m)
    del pos_error, ee_error_threshold

    if not collision_state:
        indy_dcp_cmd = np.zeros(6, dtype=np.float32)
        indy_dcp_cmd[:3] = MathFunc.m_to_mm(target_ee_pos[:3])
        roll, pitch, yaw = MathFunc.quat_to_euler_xyz(target_ee_pos[3:7])
        indy_dcp_cmd[3:] = MathFunc.rad_to_degree(np.array([roll, pitch, yaw]))
        ik_resp = robot_client.inverse_kin(indy_dcp_cmd, joint_pos_deg)
        ik_solution = ik_resp["jpos"]
        ik_state = ik_resp["response"]

        if ik_state["code"] == "0":
            joint_pos_deg = ik_solution

    return joint_pos_deg


def send_to_home(robot_client, home_joint_pos, tol=0.1):
    robot_client.movetelej_abs(home_joint_pos, vel_ratio=0.1, acc_ratio=1.0)
    while True:
        robot_state = robot_client.get_robot_data()["q"]
        joint_error = np.linalg.norm(np.array(home_joint_pos) - np.array(robot_state))
        if joint_error < tol:
            break
        time.sleep(0.1)

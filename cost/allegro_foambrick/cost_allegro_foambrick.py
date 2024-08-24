import casadi as cs
import numpy as np
import time


class Cost:
    def __init__(self, param):
        self.param_ = param

    def init_cost_fn(self):
        x = cs.SX.sym('x', self.param_.n_states_q_)
        u = cs.SX.sym('u', self.param_.n_cmd_)

        obj_pose = x[0:7]
        ftp_1_position = x[7:10]
        ftp_2_position = x[10:13]
        ftp_3_position = x[13:16]
        ftp_4_position = x[16:19]

        # target cost
        target_position = cs.SX.sym('target_position', self.param_.dims_)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        cost_param = cs.vertcat(target_position, target_quaternion)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_4_position)
        )

        # grasp cost
        ftp_1_pos_error = obj_pose[0:3] - ftp_1_position
        ftp_2_pos_error = obj_pose[0:3] - ftp_2_position
        ftp_3_pos_error = obj_pose[0:3] - ftp_3_position
        ftp_4_pos_error = obj_pose[0:3] - ftp_4_position
        grasp_cost = cs.sumsqr(ftp_1_pos_error + ftp_2_pos_error + ftp_3_pos_error + ftp_4_pos_error)

        # control cost
        control_cost = cs.sumsqr(u)

        # base cost
        base_cost = 0.0 * position_cost + 0.0 * quaternion_cost + 0 * grasp_cost + 1 * contact_cost
        final_cost = self.param_.p_weighting_ * position_cost + self.param_.q_weighting_ * quaternion_cost

        self.path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 0.2 * control_cost])
        self.final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [500 * final_cost])

        return self.path_cost_fn, self.final_cost_fn

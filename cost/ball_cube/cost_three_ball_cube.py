import casadi as cs
import numpy as np
import time
from utils import rotations

class Cost:
    def __init__(self, param):
        self.param_ = param

    def init_cost_fn(self):

        x = cs.SX.sym('x', self.param_.n_qpos_)
        u = cs.SX.sym('u', self.param_.n_cmd_)

        # target cost
        target_position = cs.SX.sym('target_position', self.param_.dims_)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        cost_param = cs.vertcat(target_position, target_quaternion)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10]) + cs.sumsqr(x[0:3] - x[10:13]) + cs.sumsqr(x[0:3] - x[13:16])
        control_cost = cs.sumsqr(u)

        # fingertip position (xy plane)
        fingertip0_xy = x[7:9]
        fingertip1_xy = x[10:12]
        fingertip2_xy = x[13:15]

        # grasp area cost function
        fingertips_area = cs.vertcat(cs.horzcat(fingertip0_xy, fingertip1_xy, fingertip2_xy), cs.DM([1, 1, 1.0]).T)
        grasp_area_cost = (cs.det(fingertips_area) ** 2 - 50e-3) ** 2

        # center cost function
        fingertips_center_xy = (fingertip0_xy + fingertip1_xy + fingertip2_xy) / 3

        # fingertip distance
        fingertips_distance = -(
                cs.sumsqr(x[7:9] - x[10:12]) + cs.sumsqr(x[13:15] - x[10:12]) + cs.sumsqr(x[7:9] - x[13:15]))

        obj_dirmat = rotations.cs_quat2rot_fn(x[3:7])
        obj_v0 = obj_dirmat.T @ (x[7:10] -  x[0:3])
        obj_v1 = obj_dirmat.T @ (x[10:13] - x[0:3])
        obj_v2 = obj_dirmat.T @ (x[13:16] - x[0:3])

        grasp_closure = cs.sumsqr(obj_v0 / (cs.sumsqr(obj_v0) + 1e-4) + obj_v1 / (cs.sumsqr(obj_v1) + 1e-4) + obj_v2 / (cs.sumsqr(obj_v2) + 1e-4))

        # grasp height
        grasp_height = cs.sumsqr(x[2] - x[9]) + cs.sumsqr(x[2] - x[12]) + cs.sumsqr(x[2] - x[15])

        # base cost
        base_cost = 0 * position_cost + 0.0 * quaternion_cost + 1 * contact_cost \
                    + 0.1 * grasp_closure + 0 * grasp_height
        final_cost = self.param_.p_weighting_ * position_cost + self.param_.q_weighting_ * quaternion_cost

        self.path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 1 * control_cost])
        self.final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [1000 * final_cost])

        self.p_q_final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [1000 * self.param_.p_weighting_ * position_cost, 1000 * self.param_.q_weighting_ * quaternion_cost])

        return self.path_cost_fn, self.final_cost_fn

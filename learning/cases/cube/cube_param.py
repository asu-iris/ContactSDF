import numpy as np

import mujoco.viewer

from utils import rotations

from utils import rotations
from cost.ball_cube.cost_three_ball_cube import Cost

class Parameters:
    def __init__(self):
        # ---------------------------------------------------------------------------------------------
        #      mode setting groups
        # ---------------------------------------------------------------------------------------------
        self.simulator_mode_ = 'balls_cube'
        self.contact_mode_ = 'lse'
        self.object_ = 'cube'

        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'env/xml/planning_mujoco_point_cube_env.xml'
            
        self.pointcloud_path_ = 'env/pc/56mm_cube.txt'

        self.h_ = 0.1
        self.frame_skip_ = int(10)

        self.robot_arm_names_ = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        obj_pose = np.array([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
        fts_qpos = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.07, -0.05, 0.0, 0.0])
        self.mj_qpos_position_ = np.hstack((obj_pose, fts_qpos))

        self.target_p_list_ = []
        self.target_p_list_.append(np.array([.05, 0.05, 0.03]))
        self.target_p_list_.append(np.array([.05, -0.05, 0.03]))
        self.target_p_list_.append(np.array([-.05, 0.05, 0.03]))
        self.target_p_list_.append(np.array([-.05, -0.05, 0.03]))

        self.target_q_list_ = []
        self.target_q_list_.append(rotations.axis_angle_to_quaternion(-1. * np.pi / 2 * np.array([0.0, .0, 1.0])))
        self.target_q_list_.append(rotations.axis_angle_to_quaternion(1. * np.pi / 2 * np.array([0.0, .0, 1.0])))
        self.target_q_list_.append(rotations.axis_angle_to_quaternion(1.0 * np.pi / 2 * np.array([0.0, 1.0, .0])))
        self.target_q_list_.append(rotations.axis_angle_to_quaternion(-1. * np.pi / 2 * np.array([0.0, 1.0, .0])))

        target_quaternion = rotations.axis_angle_to_quaternion(-1. * np.pi / 2 * np.array([0.0, 1.0, .0]))
        target_position = np.array([.1, -0.1, 0.03])
        self.target_p_ = target_position
        self.target_q_ = target_quaternion

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.con_vec_ = np.array([1.0, 1.0, 1.0])

        self.fingertip_names_ = ['fingertip0', 'fingertip1', 'fingertip2']
        self.object_names_ = ['cube']
        self.body_names_ = self.fingertip_names_ + self.object_names_

        # initialize the query point on the ground plane
        self.gd_points_ = []
        sample_start = -0.03
        sample_end = 0.03
        sample_step = 0.002
        for i in np.arange(sample_start, sample_end, sample_step):
            for j in np.arange(sample_start, sample_end, sample_step):
                self.gd_points_.append([i, j, 0.0])
        self.gd_points_ = np.array(self.gd_points_)

        self.contact_lse_smoothness_delta_ = 1e7

        # ---------------------------------------------------------------------------------------------
        #      dynamics parameters
        # ---------------------------------------------------------------------------------------------
        # system dimensions:
        self.dims_ = 3
        self.n_v_obj_ = 6

        self.ftps_n_ = 3
        self.n_qpos_ = 16
        self.n_qvel_ = 15
        self.n_cmd_ = 9
        self.njnt_trifinger_ = 9
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_

        self.max_ncon_ = 4
        
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 100

        self.mpc_u_lb_ = -0.01
        self.mpc_u_ub_ = 0.01
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(16)))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(16)))

        self.sol_guess_ = None

        self.planning_cost_ = Cost(self)
        
        self.p_weighting_ = 10
        self.q_weighting_ = 1
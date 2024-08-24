import numpy as np

import mujoco
import mujoco.viewer

from utils import rotations
from utils import rotations

from cost.allegro_stick.cost_allegro_stick import Cost

class Parameters:
    def __init__(self):
        # ---------------------------------------------------------------------------------------------
        #      mode setting groups
        # ---------------------------------------------------------------------------------------------
        self.simulator_mode_ = 'allegro'
        self.contact_mode_ = 'lse'
        self.object_ = 'stick'


        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'env/xml/planning_mujoco_allegro_stick_env.xml'


        self.pointcloud_path_ = 'env/pc/stick.txt'

        self.h_ = 0.1
        self.frame_skip_ = int(50)

        self.robot_arm_names_ = ['fingertip_0', 'fingertip_120', 'fingertip_240']

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        initial_axis_angle = 0.5 * np.pi * np.array([.0, .0, 1.0])
        initial_quaternion = rotations.axis_angle_to_quaternion(initial_axis_angle)
        obj_pose = np.hstack((np.array([0.04, -0.0, 0.025]), initial_quaternion))
        
        joint_position = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.9, 0.9, 1.3, 1.5 
        ])

        self.mj_qpos_position_ = np.hstack((joint_position, obj_pose))

        self.target_p_list_ = []
        self.target_p_list_.append(np.array([0.02, -0.0, 0.025]))
        
        self.target_q_list_ = []
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.7, 0, 0.]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.8, 0, 0.]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.3, 0, 0.]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.2, 0, 0.]) * np.pi))

        target_quaternion = rotations.rpy_to_quaternion(np.array([1.0, 0, 0.0]) * np.pi)
        target_position = np.array([-0.02, -0.0, 0.025])
        self.target_p_ = target_position
        self.target_q_ = target_quaternion

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.con_vec_ = np.array([1.0, 1.0, 1.0])

        self.fingertip_names_ = [
            'fingertip0', 'fingertip1', 'fingertip2', 'fingertip3'
        ]
        self.object_names_ = ['cube']
        self.body_names_ = self.fingertip_names_ + self.object_names_

        # initialize the query point on the ground plane
        self.gd_points_ = []
        sample_start = -0.03
        sample_end = 0.03
        sample_step = 0.005
        for i in np.arange(sample_start, sample_end, sample_step):
            for j in np.arange(sample_start, sample_end, sample_step):
                self.gd_points_.append([i, j, 0.0113])

        self.contact_margin_ = 0.01

        self.gd_points_ = np.array(self.gd_points_)

        self.contact_lse_smoothness_delta_ = 1e7

        # ---------------------------------------------------------------------------------------------
        #      dynamics parameters
        # ---------------------------------------------------------------------------------------------
        # system dimensions:
        self.dims_ = 3
        self.n_v_obj_ = 6

        self.ftps_n_ = 4
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16
        self.njnt_trifinger_ = 16
        self.n_states_q_ = 19
        self.n_states_v_ = 18

        self.max_ncon_ = 7
        
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 50

        self.mpc_jointu_lb_ = -0.1
        self.mpc_jointu_ub_ = 0.1

        fts_jointq_lb = np.array([-0.47, -0.196, -0.174, -0.227,
                                  -0.47, -0.196, -0.174, -0.227,
                                  -0.47, -0.196, -0.174, -0.227,
                                  0.263, -0.105, -0.189, -0.162 ])
        fts_jointq_ub = np.array([0.47, 1.61, 1.709, 1.618,
                                  0.47, 1.61, 1.709, 1.618,
                                  0.47, 1.61, 1.709, 1.618,
                                  1.396, 1.163, 1.70, 1.80])
        self.mpc_jointq_lb_ = np.hstack((-1e7 * np.ones(7), fts_jointq_lb))
        self.mpc_jointq_ub_ = np.hstack((1e7 * np.ones(7), fts_jointq_ub))

        self.sol_guess_ = None

        self.planning_cost_ = Cost(self)
        
        self.p_weighting_ = 1
        self.q_weighting_ = 10
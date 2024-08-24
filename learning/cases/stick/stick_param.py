import numpy as np
import mujoco.viewer
from utils import rotations

from cost.ball_stick.cost_three_ball_stick import Cost

class Parameters:
    def __init__(self):
        # ---------------------------------------------------------------------------------------------
        #      mode setting groups
        # ---------------------------------------------------------------------------------------------
        self.simulator_mode_ = 'balls_stick' 
        self.contact_mode_ = 'lse'
        self.object_ = 'stick'


        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'env/xml/planning_mujoco_point_stick_env.xml'

        self.pointcloud_path_ = 'env/pc/stick.txt'

        self.h_ = 0.1
        self.frame_skip_ = int(10)

        self.robot_arm_names_ = ['fingertip_0', 'fingertip_120', 'fingertip_240']
        self.obj_mass_ = 1e-7

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        obj_pose = np.array([0.0, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0])
        fts_qpos = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.04, 0.0, -0.05, 0.0])
        self.mj_qpos_position_ = np.hstack((obj_pose, fts_qpos))

        target_position = np.array([0.0, -0.0, 0.025])
        target_axis_angle = np.array([.5, 0.0, -0.2]) * np.pi
        target_quaternion = rotations.rpy_to_quaternion(target_axis_angle)
        self.target_p_ = target_position
        self.target_q_ = target_quaternion
        
        self.target_p_list_ = []
        self.target_p_list_.append(np.array([.05, .05, 0.01]))
        self.target_p_list_.append(np.array([.05, -.05, 0.01]))
        self.target_p_list_.append(np.array([-.05, .05, 0.01]))
        self.target_p_list_.append(np.array([-.05, -.05, 0.01]))
        self.target_q_list_ = []
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.75, 0.0, 0.0]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([1.0, 0.0, 0.0]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.0, 0.0, 0.75]) * np.pi))
        self.target_q_list_.append(rotations.rpy_to_quaternion(np.array([0.0, 0.0, 0.25]) * np.pi))

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.con_vec_ = np.array([1.0, 1.0, 1.0])

        self.fingertip_names_ = ['fingertip0', 'fingertip1', 'fingertip2']
        self.object_names_ = ['cube']
        self.body_names_ = self.fingertip_names_ + self.object_names_

        # initialize the query point on the ground plane
        self.gd_points_ = []
        for i in np.arange(-0.01, 0.01, 0.001):
            for j in np.arange(-0.01, 0.01, 0.001):
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
        fts_q_lb = np.array([-1, -1, 0.0, -1, -1, 0.0, -1, -1, 0.0])
        fts_q_ub = np.array([1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))

        self.sol_guess_ = None

        self.planning_cost_ = Cost(self)
        
        self.p_weighting_ = 10
        self.q_weighting_ = 2
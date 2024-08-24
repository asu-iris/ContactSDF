import mujoco
import mujoco.viewer

import numpy as np
import time

import dynamics.allegro_dynamics as allegro_dynamics

class MjSimulator():
    def __init__(self, param):
        
        self.param_ = param
        
        self.hand_dyn_ = allegro_dynamics.allegro_dynamics()
        
        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)
        
        self.robot_arm_names_ = ['ftp_0', 'ftp_1', 'ftp_2', 'ftp_3']
        
        self.test_ft1_cmd = np.zeros(3)
        self.keyboard_sensitivity = 0.1
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        
        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_mj_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)
    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            elif chr(keycode) == 'ĉ':
                self.test_ft1_cmd[1] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) =='Ĉ':
                self.test_ft1_cmd[1] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'ć':
                self.test_ft1_cmd[0] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ć':
                self.test_ft1_cmd[0] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'O':
                self.test_ft1_cmd[2] += 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'P':
                self.test_ft1_cmd[2] -= 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'R':
                self.test_ft1_cmd = np.array([0.0, 0.0, 0.0])
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True
        elif chr(keycode) == 'P':
            self.reset_mj_env()
    
    def reset_mj_env(self):
        self.data_.qpos[:] = self.param_.mj_qpos_position_
        self.data_.qvel[:] = np.zeros(22)
        
        mujoco.mj_forward(self.model_, self.data_)

    def initFingerJoints(self):
        for iter in range(self.param_.frame_skip_):
            self.data_.ctrl = self.param_.joint_position[0:16]
            mujoco.mj_step(self.model_, self.data_)
            time.sleep(0.001)
            self.viewer_.sync()
        
    def stepFtpsJointPositionOnce(self, fts_joints_cmd):
        desired_joint_position = fts_joints_cmd + self.get_fingertips_joint_states()
        for i in range(self.param_.frame_skip_):
            self.data_.ctrl = desired_joint_position
            mujoco.mj_step(self.model_, self.data_)
            self.viewer_.sync()
        
    def stepJointPositionOnce(self, fts_pos_cmd):
        curr_q = self.getState()
        feasible_fts_cmd = fts_pos_cmd
        
        target = (curr_q[-12:] + feasible_fts_cmd)
        e = target - self.get_fingertips_pose()
        
        for i in range(self.param_.frame_skip_):
            e = target - self.get_fingertips_pose()
            delta_q_list = []
            for i in range(0, len(self.robot_arm_names_)):
                ft_id = mujoco.mj_name2id(self.model_, mujoco.mjtObj.mjOBJ_SITE, self.robot_arm_names_[i])

                jacp = np.zeros((3, 22))
                mujoco.mj_jacSite(self.model_, self.data_, jacp, jacr=None, site=ft_id)
                
                valid_j = jacp[:, 4*i:4*i+4]
                
                inv_J_curr_ftp_part = np.linalg.pinv(valid_j)
                
                curr_ftp_error = e[3*i:3*i+3,]
                delta_q_curr_finger = inv_J_curr_ftp_part @ curr_ftp_error
                delta_q_list.append(delta_q_curr_finger)
            
            step_size = 1
            delta_q_curr_finger = np.concatenate(delta_q_list)
            control =  step_size * delta_q_curr_finger + self.data_.qpos[0:16]
            self.data_.ctrl = control
            mujoco.mj_step(self.model_, self.data_)
            self.viewer_.sync()
            e = target - self.get_fingertips_pose()
    
    def resetFingertipsPosition(self, boundary_flag):
        for iter in range(self.param_.frame_skip_):
            for i in range(boundary_flag.shape[0]):
                if boundary_flag[i] == 1:
                    self.data_.ctrl = self.param_.mj_qpos_position_[0:16]
            mujoco.mj_step(self.model_, self.data_)
            time.sleep(0.001)
            self.viewer_.sync()
            
    def get_obj_pose(self):
        obj_pos = self.data_.qpos.flatten().copy()[-7:]
        return obj_pos
    
    def get_fingertips_joint_states(self):
        joint_states = self.data_.qpos.flatten().copy()[0:-7]
        return joint_states
    
    def getPositionState(self):
        obj_pos = self.get_obj_pose()
        fts_pos = self.get_hand_fk()
        return np.concatenate((obj_pos, fts_pos))
    
    def get_hand_fk(self):
        joint_states = self.get_fingertips_joint_states()
        fts_pos = self.hand_dyn_.hand_fk_np(joint_states)
        
        return fts_pos
              
    def get_fingertips_pose(self):
        fts_pos = []
        for ft_name in self.robot_arm_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()
    
    # this is the function that gets the position of the ftps and obj
    def getState(self):
        obj_pos = self.get_obj_pose()
        ftps_pos = self.get_fingertips_joint_states()
        # ftps_pos = self.get_fingertips_pose()
        return np.concatenate((obj_pos, ftps_pos))
    
    def getQpos(self):
        return self.data_.qpos.flatten().copy()
    
    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass
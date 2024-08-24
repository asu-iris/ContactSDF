import mujoco
import mujoco.viewer

import numpy as np

import dynamics.lse_dynamics as quasiDynamic

from utils import rotations

class MjSimulator():
    def __init__(self, param):
        
        self.param_ = param
        
        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)
        
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        
        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_mj_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

        # Test
        full_inertia = np.ndarray(shape=(15, 15), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model_, full_inertia, self.data_.qM)
        obj_inertia = full_inertia[:6, :][:, :6]
        print('object inertia = ', obj_inertia)
    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True
        elif chr(keycode) == 'P':
            self.reset_obj_state()
            
    def reset_mj_env(self):
        self.data_.qpos[:] = np.copy(self.param_.mj_qpos_position_)
        self.data_.qvel[:] = np.copy(np.array(self.param_.n_qvel_ * [0]))

        mujoco.mj_forward(self.model_, self.data_)
        
    def reset_obj_state(self):
        self.data_.qpos[0:7] = np.copy(self.param_.mj_qpos_position_)[0:7]
        self.data_.qvel[:] = np.copy(np.array(self.param_.n_qvel_ * [0]))

        mujoco.mj_forward(self.model_, self.data_)
        
    def stepPositionOnce(self, fts_pos_cmd):
        curr_q = self.getState()
        feasible_fts_cmd = fts_pos_cmd

        # calculate the graviety
        fullM = np.ndarray(shape=(self.param_.n_qvel_, self.param_.n_qvel_), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model_, fullM, self.data_.qM)
        fingertipM = fullM[-self.param_.n_cmd_:, :][:, -self.param_.n_cmd_:]

        desired_fts_pos = (curr_q[-9:] + feasible_fts_cmd).copy()
        fts_dpos = []
        for _ in range(self.param_.frame_skip_):
            curr_q = self.getState()
            dpos = curr_q[-9:] - desired_fts_pos
            dvel = self.data_.qvel[-9:]
            control = -50 * dpos - 1 * dvel - fingertipM @ np.tile(self.model_.opt.gravity, 3)
            self.data_.ctrl[:] = control
            mujoco.mj_step(self.model_, self.data_, nstep=1)
            self.viewer_.sync()
            fts_dpos.append(dpos)
              
    def getState(self):
        return self.data_.qpos.flatten().copy()
    
    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass
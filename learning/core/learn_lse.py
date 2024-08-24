# -------------------------------
#      script path setting
# -------------------------------
import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(script_dir)

import casadi as cs
import numpy as np

from dynamics.lse_dynamics import LseDynamic

class LEARN_LSE():
    def __init__(self, param):
        self.param_ = param

        self.dynamics_ = LseDynamic(param)

        self.initLearning()
        self.evaFnInit()

        # Adam parameters
        self.adam_t = 1
        self.adam_vt = cs.DM.zeros(8, 1)
        self.adam_st = cs.DM.zeros(8, 1)
        self.adam_belta1 = 0.9
        self.adam_belta2 = 0.999
        self.adam_epsilon = 1e-6
        self.adam_learning_rate = 0.05

    def evaFnInit(self):
        pred_q = cs.SX.sym('pred_q', self.param_.n_qpos_)
        q_plus = cs.SX.sym('q_plus', self.param_.n_qpos_)

        q1 = pred_q[3:7] / cs.norm_2(pred_q[3:7])
        q2 = q_plus[3:7] / cs.norm_2(q_plus[3:7])

        error_trans_obj = pred_q[0:3] - q_plus[0:3]
        error_trans_ftps = pred_q[7:] - q_plus[7:]
        error_trans = cs.vertcat(error_trans_obj, error_trans_ftps)
        error_quat = 1 - cs.dot(q1, q2) ** 2
        curr_loss_t = cs.sumsqr(error_trans) * self.param_.p_weighting_
        curr_loss_r = cs.sumsqr(error_quat) * self.param_.q_weighting_

        self.evaluation_fn = cs.Function('evaluation_fn', [pred_q, q_plus], [curr_loss_t + curr_loss_r])
        
    def initLearning(self):
        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)

        contact_phi = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        jac_n = cs.SX.sym('contact_j_n', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        jac_f = cs.SX.sym('contact_j_f', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        theta = cs.SX.sym('theta', 8)

        target_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        pred_q = self.dynamics_.dyn_once_fn(curr_q, cmd, contact_phi, jac_n, jac_f, theta)

        q1 = pred_q[3:7] / cs.norm_2(pred_q[3:7])
        q2 = target_q[3:7] / cs.norm_2(target_q[3:7])
        error_trans_obj = pred_q[0:3] - target_q[0:3]
        error_trans_ftps = pred_q[7:] - target_q[7:]
        error_trans = cs.vertcat(error_trans_obj, error_trans_ftps)
        error_quat = 1 - cs.dot(q1, q2) ** 2
        loss_t = cs.sumsqr(error_trans) * self.param_.p_weighting_
        loss_r = cs.sumsqr(error_quat) * self.param_.q_weighting_
        loss = loss_t + loss_r
        
        d_loss = cs.gradient(loss, theta)

        self.loss_fn = cs.Function('loss_fn', [curr_q, cmd, contact_phi, jac_n, jac_f, target_q, theta], [loss, d_loss, loss_t, loss_r])

    def parseDataDict(self, data_list):
        curr_q_batch = []
        cmd_batch = []
        contact_phi_batch = []
        jac_n_batch = []
        jac_f_batch = []
        target_q_batch = []
        for data in data_list:
            curr_q_batch.append(data['q'])
            cmd_batch.append(data['u'])
            contact_phi_batch.append(data['phi'])
            jac_n_batch.append(data['jac_n'])
            jac_f_batch.append(data['jac_f'])
            target_q_batch.append(data['q_plus'])

        curr_q_batch = np.stack(curr_q_batch, axis=-1)
        cmd_batch = np.stack(cmd_batch, axis=-1)
        contact_phi_batch = np.stack(contact_phi_batch, axis=-1)
        jac_n_batch = np.hstack(jac_n_batch)
        jac_f_batch = np.hstack(jac_f_batch)
        target_q_batch = np.stack(target_q_batch, axis=-1)
    
        return curr_q_batch, cmd_batch, contact_phi_batch, jac_n_batch, jac_f_batch, target_q_batch
    
    def adamGradient(self, vt, st, gt, t):
        updated_vt = self.adam_belta1 * vt + (1 - self.adam_belta1) * (gt)
        updated_st = self.adam_belta2 * st + (1 - self.adam_belta2) * (gt**2)
        updated_t = t + 1
        
        v_bias_corr = updated_vt / (1 - self.adam_belta1**t)
        s_bias_corr = updated_st / (1 - self.adam_belta2**t)
        corrected_grad = self.adam_learning_rate * v_bias_corr / (np.sqrt(s_bias_corr) + self.adam_epsilon)
        
        return updated_vt, updated_st, updated_t, corrected_grad

    def batchLearnOnce(self, data_list, theta):
        curr_q_batch, cmd_batch, contact_phi_batch, jac_n_batch, jac_f_batch, target_q_batch = self.parseDataDict(data_list)

        loss, d_loss, loss_t, loss_r = self.loss_fn(curr_q_batch, cmd_batch, contact_phi_batch, jac_n_batch, jac_f_batch, target_q_batch, theta)

        loss = np.mean(loss, axis=-1)
        d_loss = np.mean(d_loss, axis=-1)
        loss_t = np.mean(loss_t, axis=-1)
        loss_r = np.mean(loss_r, axis=-1)

        updated_vt, updated_st, updated_t, corr_grad = self.adamGradient(self.adam_vt, self.adam_st, d_loss, self.adam_t)

        theta_plus = theta - corr_grad
        self.adam_vt = updated_vt
        self.adam_st = updated_st
        self.adam_t = updated_t

        return theta_plus

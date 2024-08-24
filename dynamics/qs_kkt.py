import mujoco
import mujoco.viewer

import numpy as np

np.set_printoptions(suppress=True)
import casadi as cs

class QsKKT:
    def __init__(self, param):
        self.param_ = param

        self.qsKKTInit()
        
        self.qsDynamicsInit()
        
        self.initUtilsFunctions()
        
    
    def qsKKTInit(self):
        object_vel = cs.SX.sym('object_vel', self.param_.n_v_obj_)
        fingertips_vel = cs.SX.sym('fingertips_vel', self.param_.n_cmd_)
        
        contact_distances = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        contact_jacobians = cs.SX.sym('contact_jacobians', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        
        vel = cs.vertcat(object_vel, fingertips_vel)
        u = cs.SX.sym('u', self.param_.n_cmd_)
        
        inertial_vec = cs.vertcat(cs.repmat(self.param_.M_tran, 3), self.param_.M_rotation)
        inertial_mat = cs.diag(inertial_vec)
        K_stiff_vec = cs.repmat(self.param_.K_stiff, self.param_.n_cmd_)
        K_stiff_mat = cs.diag(K_stiff_vec)
        
        # external forces on the cube
        tau_object = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        tau_fingertips = K_stiff_mat @ u
        tau = cs.vertcat(tau_object, tau_fingertips)
        
        qp_cost = 0.5 * cs.dot(inertial_mat @ object_vel, object_vel) + \
                  0.5 * cs.dot(K_stiff_mat @ fingertips_vel, fingertips_vel) - cs.dot(tau, vel) / self.param_.h_

        qp_g = contact_distances / self.param_.h_ + contact_jacobians @ vel

        self.qs_model_dict_ = dict(qp_cost_fn=cs.Function('qp_cost_fn', [vel, u], [qp_cost]),
                                        qp_g_fn=cs.Function('qp_g_fn', [vel, contact_distances, contact_jacobians],
                                                            [qp_g])) 

    def initUtilsFunctions(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', self.param_.n_qvel_)
        qpos = cs.SX.sym('qpos', self.param_.n_qpos_)
        next_obj_pos = qpos[0:3] + self.param_.h_ * qvel[0:3]
        next_fts_pos = qpos[7:] + self.param_.h_ * qvel[6:]
        next_obj_quat = (qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        # next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat, next_fts_pos)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])



    def qsDynamicsInit(self):
        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)
        contact_distances = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        contact_jacobians = cs.SX.sym('contact_jacobians', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        object_vel = cs.SX.sym('object_vel', self.param_.n_v_obj_)
        fingertips_vel = cs.SX.sym('fingertips_vel', self.param_.n_cmd_)

        vel = cs.vertcat(object_vel, fingertips_vel)
        
        # external forces on the cube
        tau_object = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        tau_fingertips = self.param_.fts_stiff_ @ cmd
        tau = cs.vertcat(tau_object, tau_fingertips)
        
        qp_objective = 0.5 * self.param_.h_ * self.param_.h_ * cs.dot(self.param_.obj_inertia_ @ object_vel, object_vel) + \
                       0.5 * self.param_.h_ * self.param_.h_ * cs.dot(self.param_.fts_stiff_ @ fingertips_vel, fingertips_vel) - \
                       self.param_.h_ * cs.dot(tau, vel)
        
        qp_constraints = contact_distances / self.param_.h_ + contact_jacobians @ vel
        
        cs_param = cs.vertcat(cmd, cs.vec(contact_distances), cs.vec(contact_jacobians))

        quadprog = {'x': vel, 'f': qp_objective, 'g': qp_constraints, 'p': cs_param}

        opts = {'error_on_fail': False, 'printLevel': 'none'}
        self.qp_solver_fn_ = cs.qpsol('qp_solver', 'qpoases', quadprog, opts)
        
        # opts = {'error_on_fail': False}
        # self.qp_solver_fn_ = cs.qpsol('qp_solver', 'proxqp', quadprog, opts)

        # self.qp_solver_fn_ = cs.qpsol('qp_solver', 'proxqp', quadprog, opts)

        
    def dyn_fn(self, curr_q, cmd, contact_distances, contact_jacobians): 
        qp_param = cs.vertcat(cmd, cs.vec(contact_distances), cs.vec(contact_jacobians))
        sol = self.qp_solver_fn_(p=qp_param, lbg=0.0)
        v = sol['x'].full().flatten()
        
        next_qpos = self.cs_qposInteg_(curr_q, v)

        return next_qpos
    
    def dyn_vel_fn(self, curr_q, cmd, contact_distances, contact_jacobians): 
        qp_param = cs.vertcat(cmd, cs.vec(contact_distances), cs.vec(contact_jacobians))
        sol = self.qp_solver_fn_(p=qp_param, lbg=0.0)
        v = sol['x'].full().flatten()
        
        return v
    
    def dyn_distance_fn(self, curr_q, cmd, contact_distances, contact_jacobians): 
        qp_param = cs.vertcat(cmd, cs.vec(contact_distances), cs.vec(contact_jacobians))
        sol = self.qp_solver_fn_(p=qp_param, lbg=0.0)
        v = sol['x'].full().flatten()
        
        Q = np.zeros((15, 15))
        Q[:6, :6] = self.param_.obj_inertia_
        Q[6:15, 6:15] = self.param_.fts_stiff_
        
        b_o = self.param_.obj_mass_ * self.param_.gravity_
        b_r = self.param_.fts_stiff_ @ cmd
        b_u = np.hstack((b_o, b_r))
        
        z_opt = self.param_.h_ * np.sqrt(Q) @ v 
        z_query = np.linalg.inv(np.sqrt(Q)) @ b_u
        
        return np.linalg.norm(z_opt - z_query)
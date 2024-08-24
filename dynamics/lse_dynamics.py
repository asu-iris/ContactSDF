import mujoco
import mujoco.viewer

import numpy as np

np.set_printoptions(suppress=True)
import casadi as cs

class LseDynamic:
    def __init__(self, param):
        self.param_ = param

        self.initUtilsFunctions()
        self.D_SDFInit()
        self.initLSEDynamics()

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

    def initLSEDynamics(self):

        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)

        theta = cs.SX.sym('theta', 8)
        sigma = theta[0]
        M_tran_recip_sqrt = theta[1]
        M_rotation_recip_sqrt = theta[2:5]
        K_stiff_recip_sqrt = theta[5]
        mass_sqrt = theta[6]
        friction_mu = theta[7]

        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_n = cs.SX.sym('jac_n', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        jac_f = cs.SX.sym('jac_f', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        jac_mat = jac_n + friction_mu * jac_f

        K_stiff_scale = 1 / K_stiff_recip_sqrt**2
        mass = mass_sqrt**2

        # vectors
        M_trans_recip_sqrt_vec = cs.repmat(M_tran_recip_sqrt, 3, 1)
        M_rotation_recip_sqrt_vec = M_rotation_recip_sqrt
        M_recip_sqrt_vec = cs.vertcat(M_trans_recip_sqrt_vec, M_rotation_recip_sqrt_vec)

        K_recip_sqrt_vec = cs.repmat(K_stiff_recip_sqrt, self.param_.n_cmd_, 1)
        
        # matrices
        K_mat = cs.diag(cs.repmat(K_stiff_scale, self.param_.n_cmd_, 1))
        Q_sqrt_inv = cs.diag(cs.vertcat(M_recip_sqrt_vec, K_recip_sqrt_vec))

        gravity = cs.DM(self.param_.gravity_)
        b_obj = mass * gravity
        b_fts = K_mat @ cmd
        b_u = cs.vertcat(b_obj, b_fts)

        z_query = Q_sqrt_inv @ b_u

        cs_param = cs.vertcat(
            z_query, cs.vec(phi_vec), cs.vec(jac_mat), 
            cs.vec(sigma), cs.vec(Q_sqrt_inv)
        )

        approx_optimal, argmin_gradient = self.dynamics_lse_fn_(cs_param)
        optimal_z = z_query - approx_optimal @ argmin_gradient
        optimal_v = Q_sqrt_inv @ optimal_z / self.param_.h_
        v = optimal_v
        
        next_qpos = self.cs_qposInteg_(curr_q, v)

        self.dyn_once_fn = cs.Function('dyn_once_fn', [curr_q, cmd, phi_vec, jac_n, jac_f, theta], [next_qpos])

    def D_SDFInit(self):
        # set optimization variable
        cs_query_z = cs.SX.sym('cs_query_z', self.param_.n_qvel_)
        cs_sigma = cs.SX.sym('cs_sigma', 1)
        
        Q_inv_sqrt = cs.SX.sym('Q_inv_sqrt', self.param_.n_qvel_, self.param_.n_qvel_)

        # constraints
        cs_phi_vec = cs.SX.sym('cs_phi_vec', self.param_.max_ncon_ * 4)
        cs_jac_mat = cs.SX.sym('cs_jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        cs_A = cs_jac_mat @ Q_inv_sqrt / self.param_.h_
        cs_b = cs_phi_vec / self.param_.h_

        row_normer = cs.sqrt(cs.inv(cs.diag(cs.diag(cs_A @ cs_A.T) + 1e-2)))
        cs_A = row_normer @ cs_A
        cs_b = row_normer @ cs_b

        values = - cs_b - cs_A @ cs_query_z
        distance = values * cs_sigma

        # -------------------------------
        #         casadi new lse 
        # -------------------------------
        logsumexp_result = cs.logsumexp(distance)

        max_approx = logsumexp_result / cs_sigma

        max_approx = cs.fmax(0, max_approx)

        dyn_gradient_z = cs.gradient(max_approx, cs_query_z)

        cs_param = cs.vertcat(
            cs_query_z, cs.vec(cs_phi_vec), cs.vec(cs_jac_mat), 
            cs_sigma, cs.vec(Q_inv_sqrt)
        )
        self.dynamics_lse_fn_ = cs.Function('dynamics_lse_function', [cs_param], [max_approx, dyn_gradient_z])


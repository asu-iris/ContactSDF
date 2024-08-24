import casadi as cs
import numpy as np
import time

import dynamics.allegro_dynamics as allegro_dynamics 

from dynamics.joint_lse_dynamics import LseDynamic

class MPC_IPOPT:
    def __init__(self, param):
        self.param_ = param
        
        self.hand_dyn = allegro_dynamics.allegro_dynamics()
        
        task_cost = self.param_.planning_cost_
        self.path_cost_fn, self.final_cost_fn = task_cost.init_cost_fn()
        
        self.mpc_mode_ = 'lse'
        
        if self.mpc_mode_ == 'lse':
            self.dynamics_ = LseDynamic(param)
            self.initLSEMpc()
            self.initThetaFn()

    def initThetaFn(self):

        sigma = cs.SX.sym('sigma', 1)
        M_tran = cs.SX.sym('M_tran', 1)
        M_rotation = cs.SX.sym('M_rotation', 3)
        K_stiff = cs.SX.sym('K_stiff', 1)
        mass = cs.SX.sym('mass', 1)
        friction_mu = cs.SX.sym('friction_mu', 1)

        M_tran_recip_sqrt = 1 / cs.sqrt(M_tran)
        M_rotation_recip_sqrt = 1 / cs.sqrt(M_rotation)
        K_stiff_recip_sqrt = 1 / cs.sqrt(K_stiff)
        mass_sqrt = cs.sqrt(mass)
        theta = cs.vvcat([sigma, M_tran_recip_sqrt, M_rotation_recip_sqrt, K_stiff_recip_sqrt, mass_sqrt, friction_mu])

        self.get_theta_fn_ = cs.Function('get_theta_fn', 
            [sigma, M_tran, M_rotation, K_stiff, mass, friction_mu], [theta])

        return theta
        
    def planOnce(self, 
            target_p, target_q, 
            curr_x, contact_distances, 
            contact_j_n, contact_j_f,  
            theta,
            sol_guess=None):

        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        cost_params = cs.vertcat(target_p, target_q)
        
        nlp_param = self.nlp_params_fn_(curr_x, contact_distances, contact_j_n, contact_j_f, cost_params, theta)
        
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(
            self.param_.mpc_jointu_lb_, self.param_.mpc_jointu_ub_, 
            self.param_.mpc_jointq_lb_, self.param_.mpc_jointq_ub_
        )

        st = time.time()
        raw_sol = self.ipopt_solver(
            x0=sol_guess['x0'],
            lam_x0=sol_guess['lam_x0'],
            lam_g0=sol_guess['lam_g0'],
            lbx=nlp_lbw, ubx=nlp_ubw,
            lbg=0.0, ubg=0.0,
            p=nlp_param
        )
        
        w_opt = raw_sol['x'].full().flatten()
        cost_opt = raw_sol['f'].full().flatten()
        
        # extract the solution from the raw solution
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, -1))
        opt_u_traj = sol_traj[:, 0:self.param_.njnt_trifinger_]
        opt_pred_q = sol_traj[:, self.param_.n_cmd_:]
        
        print("mpc solving time:", (time.time() - st) * 1000)
        print('return_status = ', self.ipopt_solver.stats()['return_status'])
        
        return dict(action=opt_u_traj[0, :],
                    prediction=opt_pred_q[0, :],
                    sol_guess=dict(x0=w_opt,
                                   lam_x0=raw_sol['lam_x'],
                                   lam_g0=raw_sol['lam_g'],
                                   opt_cost=raw_sol['f'].full().item()),
                    cost_opt=cost_opt,
                    solve_status=self.ipopt_solver.stats()['return_status'])
        
    def initLSEMpc(self):
        lse_sigma = cs.SX.sym('lse_sigma', 1)

        contact_distances = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        contact_jacobians = cs.SX.sym('contact_jacobians', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        sigma = cs.SX.sym('sigma', 1)
        M_tran_recip_sqrt = cs.SX.sym('M_tran_recip_sqrt', 1)
        M_rotation_recip_sqrt = cs.SX.sym('M_rotation_recip_sqrt', 3)
        K_stiff_recip_sqrt = cs.SX.sym('K_stiff_recip_sqrt', 1)
        mass_sqrt = cs.SX.sym('mass_sqrt', 1)
        friction_mu = cs.SX.sym('friction_mu', 1)
        theta = cs.vvcat([sigma, M_tran_recip_sqrt, M_rotation_recip_sqrt, K_stiff_recip_sqrt, mass_sqrt, friction_mu])

        contact_distances = cs.SX.sym('contact_distances', self.param_.max_ncon_ * 4)
        contact_j_n = cs.SX.sym('contact_j_n', self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        contact_j_f = cs.SX.sym('contact_j_f', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        cost_params = cs.SX.sym('cost_params', self.path_cost_fn.size_in(2))
        lbu = cs.SX.sym('lbu', self.param_.njnt_trifinger_)
        ubu = cs.SX.sym('ubu', self.param_.njnt_trifinger_)
        
        lbq = cs.SX.sym('lbq', self.param_.n_qpos_)
        ubq = cs.SX.sym('ubq', self.param_.n_qpos_)
        
        # start with empty NLP
        w, w0, lbw, ubw, g = [], [], [], [], []
        J = 0.0
        q0 = cs.SX.sym('q0', self.param_.n_qpos_)
        qk = q0
        ef_pos_0 = self.hand_dyn.hand_fk_cs(q0[7:])
        x0 = cs.vertcat(q0[0:7], ef_pos_0)
        xk = x0
        for k in range(self.param_.mpc_horizon_):
            # control at time k
            uk = cs.SX.sym('u' + str(k), self.param_.njnt_trifinger_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.njnt_trifinger_)]

            # lse dyn function
            pred_q = self.dynamics_.dyn_once_fn(
                qk, uk, contact_distances, contact_j_n, contact_j_f, theta
            )

            # compute the cost function
            J += self.path_cost_fn(xk, uk, cost_params)

            # q at time k+1 .... q_new
            qk = cs.SX.sym('q' + str(k + 1), self.param_.n_qpos_)
            ef_pos_k = self.hand_dyn.hand_fk_cs(qk[7:])
            xk = cs.vertcat(qk[0:7], ef_pos_k)
            
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]

            # add the concatenation constraint
            g += [pred_q - qk]

        # compute the final cost
        J += self.final_cost_fn(xk, cost_params)
        
        # create an NLP solver
        nlp_params = cs.vvcat([
            q0, contact_distances, contact_j_n, contact_j_f, cost_params, theta])
        nlp_prog = {'f': J, 'x': cs.vcat(w), 'g': cs.vcat(g), 'p': nlp_params}
        nlp_opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
                    'ipopt.max_iter': self.param_.ipopt_max_iter_}
        self.ipopt_solver = cs.nlpsol('solver', 'ipopt', nlp_prog, nlp_opts)

        # useful mappings
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function('nlp_bounds_fn', [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function('nlp_params_fn',
            [q0, contact_distances, contact_j_n, contact_j_f, cost_params, theta], 
            [nlp_params])


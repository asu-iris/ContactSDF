# -------------------------------
#      script path setting
# -------------------------------
import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
sys.path.append(script_dir)

from learning.cases.stick.stick_param import Parameters
param = Parameters()

from simulator.fts_simulator import MjSimulator

if param.contact_mode_ == 'lse':
    from contact.ball_lse_contact_detection import Contact
    
from planning.ball_mpc_lse import MPC_IPOPT

from learning.core.learn_lse import LEARN_LSE

import random
import numpy as np

# -------------------------------
#        init contact
# -------------------------------
contact = Contact(param)

# -------------------------------
#        init simulator
# -------------------------------
simulator = MjSimulator(param)

# -------------------------------
#        init planner
# -------------------------------
planner = MPC_IPOPT(param)

# -------------------------------
#      init learning module
# -------------------------------
online_learning = LEARN_LSE(param)

# -------------------------------
#      init parameters
# -------------------------------
param.dyn_lse_smoothness_delta_ = 100
param.M_tran = 1
param.M_rotation = np.array([1, 1, 1])
param.K_stiff = 1
param.obj_mass_ = 0.01
param.mu_fingertip_ = 0.5
theta = planner.get_theta_fn_(
    param.dyn_lse_smoothness_delta_,
    param.M_tran,
    param.M_rotation,
    param.K_stiff,
    param.obj_mass_,
    param.mu_fingertip_
)
learn_counter = 0
while learn_counter < 20:
    learn_counter += 1
    epoch_buffer = []
    task_counter = 0
    accumulated_trans_loss = 0
    accumulated_rotation_loss = 0
    for task_id in range(len(param.target_p_list_)):
        task_p = param.target_p_list_[task_id]
        task_q = param.target_q_list_[task_id]
        simulator.set_goal(task_p, task_q)
        param.sol_guess_ = None
        print('task_counter = ', task_counter)
        task_counter += 1
        step_counter = 0
        while step_counter < 100:
            if not simulator.dyn_paused_:
            
                # get state
                curr_q = simulator.getState()

                # -----------------------
                #     contact detect
                # -----------------------
                phi_results, jac_n, jac_f = contact.detectOnce(curr_q, simulator)

                # -----------------------
                #        planning
                # -----------------------
                sol = planner.planOnce(
                    task_p,
                    task_q,
                    curr_q,
                    phi_results,
                    jac_n, jac_f,
                    theta,
                    param.sol_guess_)
                param.sol_guess_ = sol['sol_guess']
                mpc_planning = sol['action']

                # -----------------------
                #        simulate
                # -----------------------
                simulator.stepPositionOnce(mpc_planning)
                
                # -----------------------
                #    data preparation
                # -----------------------
                trajectory_data = dict(
                    q = curr_q,
                    u = mpc_planning,
                    jac_n = jac_n, jac_f = jac_f,
                    phi = phi_results,
                    q_plus = simulator.getState()
                )
                epoch_buffer.append(trajectory_data)
                step_counter += 1
        simulator.reset_mj_env()
            
    # -----------------------
    #    online learning
    # -----------------------
    random.shuffle(epoch_buffer)
    buffer_size = len(epoch_buffer)
    batch_size = 100
    for i in range(0, buffer_size, batch_size):
        batch_data = epoch_buffer[i:i+batch_size]

        theta_plus = online_learning.batchLearnOnce(batch_data, theta)
        theta = theta_plus
        print('theta = ', theta)
        
    # exit signal
    if simulator.break_out_signal_:
        exit()
import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
sys.path.append(script_dir)

# -------------------------------
#        init parameters
# -------------------------------
from learning.cases.foambrick.foam_param import Parameters
param = Parameters()

from simulator.fts_simulator import MjSimulator

if param.contact_mode_ == 'lse':
    from contact.ball_lse_contact_detection import Contact

from planning.ball_mpc_lse import MPC_IPOPT

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

theta = np.array([120.223, 0.403609, 1.5038, 1.14279, 1.55258, 0.99718, -0.015059, 0.340951])

for task_id in range(len(param.target_p_list_)):
    task_p = param.target_p_list_[task_id]
    task_q = param.target_q_list_[task_id]
    simulator.set_goal(task_p, task_q)
    param.sol_guess_ = None
    step_counter = 0
    while step_counter < 300:
        if not simulator.dyn_paused_:
            print('--------------------------------')
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

            # exit signal
            if simulator.break_out_signal_:
                os._exit()
            step_counter += 1
    simulator.reset_mj_env()

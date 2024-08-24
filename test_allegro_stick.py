import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
sys.path.append(script_dir)

# -------------------------------
#        init parameters
# -------------------------------
from learning.cases.allegro_stick.allegrostick_param import Parameters
param = Parameters()

from simulator.allegro_simulator import MjSimulator

if param.contact_mode_ == 'lse':
    from contact.sim_joint_lse_contact_detection import Contact
    
from planning.hand_mpc_lse import MPC_IPOPT

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

theta = np.array([99.6527, 0.651158, 0.712172, 0.65902, 0.668154, 1.26332, -0.0468192, 0.155784])

for task_p in param.target_p_list_:
    for task_q in param.target_q_list_:
        simulator.set_goal(np.array([-0.15,0,0.04]), task_q)
        param.sol_guess_ = None
        step_counter = 0
        while step_counter < 200:
            if not simulator.dyn_paused_:
                print('--------------------------------')
                # get state
                curr_q = simulator.getState()
                curr_position_state = simulator.getPositionState()

                # -----------------------
                #     contact detect
                # -----------------------
                phi_results, jac_n, jac_f = contact.detectOnce(curr_position_state, curr_q, simulator)

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
                simulator.stepFtpsJointPositionOnce(mpc_planning)

                # exit signal
                if simulator.break_out_signal_:
                    os._exit()
                step_counter += 1
        simulator.reset_mj_env()

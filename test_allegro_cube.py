import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
sys.path.append(script_dir)

# -------------------------------
#        init parameters
# -------------------------------
from learning.cases.allegro_cube.allegrocube_param import Parameters
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

theta = np.array([119.659, 0.653641, 0.875519, 0.766121, 0.670051, 1.3007, 0.0913508, 0.173678])

# while simulator.viewer_.is_running():
for task_p in param.target_p_list_:
    for task_q in param.target_q_list_:
        simulator.set_goal(np.array([-0.15,0,0.04]), task_q)
        param.sol_guess_ = None
        step_counter = 0
        while step_counter < 150:
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
                joint_delta_cmd = sol['action']

                # -----------------------
                #        simulate
                # -----------------------
                simulator.stepFtpsJointPositionOnce(joint_delta_cmd)

                # exit signal
                if simulator.break_out_signal_:
                    os._exit()
                step_counter += 1
        simulator.reset_mj_env()

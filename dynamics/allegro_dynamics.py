import casadi as cs
import numpy as np

class allegro_dynamics:
    def __init__(self):
        self.initFunctions()
        self.initAllegroFK()
        self.initAllegroJ()

    def initFunctions(self):
        pos = cs.SX.sym('pos', 3)
        ttmat = cs.vertcat(
            cs.horzcat(1, 0, 0, pos[0]),
            cs.horzcat(0, 1, 0, pos[1]),
            cs.horzcat(0, 0, 1, pos[2]),
            cs.horzcat(0, 0, 0, 1)
        )
        self.ttmat_fn = cs.Function('ttmat_fn', [pos], [ttmat])

        # ----------------
        # rotations to homogeneous transformation mat (casadi_fn)
        alpha = cs.SX.sym('alpha', 1)
        rxtmat = cs.vertcat(
            cs.horzcat(1, 0, 0, 0),
            cs.horzcat(0, cs.cos(alpha), -cs.sin(alpha), 0),
            cs.horzcat(0, cs.sin(alpha), cs.cos(alpha), 0),
            cs.horzcat(0, 0, 0, 1)
        )
        self.rxtmat_fn = cs.Function('rxtmat_fn', [alpha], [rxtmat])

        beta = cs.SX.sym('beta', 1)
        rytmat = cs.vertcat(
            cs.horzcat(cs.cos(beta), 0, cs.sin(beta), 0),
            cs.horzcat(0, 1, 0, 0),
            cs.horzcat(-cs.sin(beta), 0, cs.cos(beta), 0),
            cs.horzcat(0, 0, 0, 1)
        )
        self.rytmat_fn = cs.Function('rytmat_fn', [beta], [rytmat])

        theta = cs.SX.sym('theta', 1)
        rztmat = cs.vertcat(
            cs.horzcat(cs.cos(theta), -cs.sin(theta), 0, 0),
            cs.horzcat(cs.sin(theta), cs.cos(theta), 0, 0),
            cs.horzcat(0, 0, 1, 0),
            cs.horzcat(0, 0, 0, 1),
        )
        self.rztmat_fn = cs.Function('rztmat_fn', [theta], [rztmat])
        
        quat = cs.SX.sym('quat', 4)
        # Extract the values from Q
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 4x4 rotation matrix
        quat_tmat = np.array([[r00, r01, r02, 0],
                            [r10, r11, r12, 0],
                            [r20, r21, r22, 0],
                            [0, 0, 0, 1]])
        

        self.quattmat_fn = cs.Function('quattmat_fn', [quat], [quat_tmat])
        
    def initAllegroFK(self):
        # palm
        t_palm = self.quattmat_fn(np.array([0, 1, 0, 1]) / np.linalg.norm([0, 1, 0, 1]))
        
        ff_mf_rf_tip_link_length = 0.0267
        th_tip_link_length = 0.0423

        # first finger
        ff_qpos = cs.SX.sym('ff_qpos', 4)
        ff_t_base = t_palm @ self.ttmat_fn([0, 0.0435, -0.001542]) @ self.quattmat_fn([0.999048, -0.0436194, 0, 0])
        ff_t_proximal = ff_t_base @ self.rztmat_fn(ff_qpos[0]) @ self.ttmat_fn([0, 0, 0.0164])
        ff_t_medial = ff_t_proximal @ self.rytmat_fn(ff_qpos[1]) @ self.ttmat_fn([0, 0, 0.054])
        ff_t_distal = ff_t_medial @ self.rytmat_fn(ff_qpos[2]) @ self.ttmat_fn([0, 0, 0.0384])
        ff_t_ftp = ff_t_distal @ self.rytmat_fn(ff_qpos[3]) @ self.ttmat_fn([0, 0, ff_mf_rf_tip_link_length])
        self.fftp_pos_fd_fn = cs.Function('ff_t_ftp_fn', [ff_qpos], [ff_t_ftp[0:3, -1]])

        # middle finger
        mf_qpos = cs.SX.sym('mf_qpos', 4)
        mf_t_base = t_palm @ self.ttmat_fn([0, 0, 0.0007])
        mf_t_proximal = mf_t_base @ self.rztmat_fn(mf_qpos[0]) @ self.ttmat_fn([0, 0, 0.0164])
        mf_t_medial = mf_t_proximal @ self.rytmat_fn(mf_qpos[1]) @ self.ttmat_fn([0, 0, 0.054])
        mf_t_distal = mf_t_medial @ self.rytmat_fn(mf_qpos[2]) @ self.ttmat_fn([0, 0, 0.0384])
        mf_t_ftp = mf_t_distal @ self.rytmat_fn(mf_qpos[3]) @ self.ttmat_fn([0, 0, ff_mf_rf_tip_link_length])
        self.mftp_pos_fd_fn = cs.Function('mftp_pos_fd_fn', [mf_qpos], [mf_t_ftp[0:3, -1]])

        # ring finger
        rf_qpos = cs.SX.sym('rf_qpos', 4)
        rf_t_base = t_palm @ self.ttmat_fn([0, -0.0435, -0.001542]) @ self.quattmat_fn([0.999048, 0.0436194, 0, 0])
        rf_t_proximal = rf_t_base @ self.rztmat_fn(rf_qpos[0]) @ self.ttmat_fn([0, 0, 0.0164])
        rf_t_medial = rf_t_proximal @ self.rytmat_fn(rf_qpos[1]) @ self.ttmat_fn([0, 0, 0.054])
        rf_t_distal = rf_t_medial @ self.rytmat_fn(rf_qpos[2]) @ self.ttmat_fn([0, 0, 0.0384])
        rf_t_ftp = rf_t_distal @ self.rytmat_fn(rf_qpos[3]) @ self.ttmat_fn([0, 0, ff_mf_rf_tip_link_length])
        self.rftp_pos_fd_fn = cs.Function('rftp_pos_fd_fn', [rf_qpos], [rf_t_ftp[0:3, -1]])

        # Thumb
        th_qpos = cs.SX.sym('th_qpos', 4)
        th_t_base = t_palm @ self.ttmat_fn([-0.0182, 0.019333, -0.045987]) @ self.quattmat_fn([0.477714, -0.521334, -0.521334, -0.477714])
        th_t_proximal = th_t_base @ self.rxtmat_fn(-th_qpos[0]) @ self.ttmat_fn([-0.027, 0.005, 0.0399])
        th_t_medial = th_t_proximal @ self.rztmat_fn(th_qpos[1]) @ self.ttmat_fn([0, 0, 0.0177])
        th_t_distal = th_t_medial @ self.rytmat_fn(th_qpos[2]) @ self.ttmat_fn([0, 0, 0.0514])
        th_t_ftp = th_t_distal @ self.rytmat_fn(th_qpos[3]) @ self.ttmat_fn([0, 0, th_tip_link_length])
        self.thtp_pos_fd_fn = cs.Function('thtp_pos_fd_fn', [th_qpos], [th_t_ftp[0:3, -1]])

    def hand_fk_np(self, ftp_joint_states):
        ff_joints = ftp_joint_states[0:4]
        mf_joints = ftp_joint_states[4:8]
        rf_joints = ftp_joint_states[8:12]
        th_joints = ftp_joint_states[12:16]
        
        ff_position = self.fftp_pos_fd_fn(ff_joints)
        mf_position = self.mftp_pos_fd_fn(mf_joints)
        rf_position = self.rftp_pos_fd_fn(rf_joints)
        th_position = self.thtp_pos_fd_fn(th_joints)
        
        return np.vstack((ff_position, mf_position, rf_position, th_position)).flatten().copy()
    
    def hand_fk_cs(self, ftp_joint_states):
        ff_joints = ftp_joint_states[0:4]
        mf_joints = ftp_joint_states[4:8]
        rf_joints = ftp_joint_states[8:12]
        th_joints = ftp_joint_states[12:16]
        
        ff_position = self.fftp_pos_fd_fn(ff_joints)
        mf_position = self.mftp_pos_fd_fn(mf_joints)
        rf_position = self.rftp_pos_fd_fn(rf_joints)
        th_position = self.thtp_pos_fd_fn(th_joints)
        
        return cs.vcat((ff_position, mf_position, rf_position, th_position))
    
    def initAllegroJ(self):
        # palm
        t_palm = self.quattmat_fn(np.array([0, 1, 0, 1]) / np.linalg.norm([0, 1, 0, 1]))
        
        ff_qpos = cs.SX.sym('ff_qpos', 4)
        mf_qpos = cs.SX.sym('mf_qpos', 4)
        rf_qpos = cs.SX.sym('rf_qpos', 4)

        f3_proximal_t = cs.SX.sym('f3_proximal_t', 3)
        f3_medial_t = cs.SX.sym('f3_medial_t', 3)
        f3_distal_t = cs.SX.sym('f3_distal_t', 3)
        f3_tip_t = cs.SX.sym('f3_tip_t', 3)
        f3_t_param = cs.vvcat([f3_proximal_t, f3_medial_t, f3_distal_t, f3_tip_t])

        th_qpos = cs.SX.sym('th_qpos', 4)
        th_proximal_t = cs.SX.sym('th_proximal_t', 3)
        th_medial_t = cs.SX.sym('th_medial_t', 3)
        th_distal_t = cs.SX.sym('th_distal_t', 3)
        th_tip_t = cs.SX.sym('th_tip_t', 3)
        th_t_param = cs.vvcat([th_proximal_t, th_medial_t, th_distal_t, th_tip_t])

        q_pos = cs.vvcat([ff_qpos, mf_qpos, rf_qpos, th_qpos])

        # first finger
        ff_input_param = cs.vvcat([ff_qpos, f3_t_param])
        ff_t_base = t_palm @ self.ttmat_fn([0, 0.0435, -0.001542]) @ self.quattmat_fn([0.999048, -0.0436194, 0, 0])
        ff_t_proximal = ff_t_base @ self.rztmat_fn(ff_qpos[0]) @ self.ttmat_fn(f3_proximal_t)
        ff_t_medial = ff_t_proximal @ self.rytmat_fn(ff_qpos[1]) @ self.ttmat_fn(f3_medial_t)
        ff_t_distal = ff_t_medial @ self.rytmat_fn(ff_qpos[2]) @ self.ttmat_fn(f3_distal_t)
        ff_t_ftp = ff_t_distal @ self.rytmat_fn(ff_qpos[3]) @ self.ttmat_fn(f3_tip_t)
        ff_jacp = cs.jacobian(ff_t_ftp[0:3, -1], q_pos)
        self.ff_J_fn = cs.Function('ff_J_fn', [ff_input_param], [ff_jacp, ff_t_ftp[0:3, -1]])

        # middle finger
        mf_input_param = cs.vvcat([mf_qpos, f3_t_param])
        mf_t_base = t_palm @ self.ttmat_fn([0, 0, 0.0007])
        mf_t_proximal = mf_t_base @ self.rztmat_fn(mf_qpos[0]) @ self.ttmat_fn(f3_proximal_t)
        mf_t_medial = mf_t_proximal @ self.rytmat_fn(mf_qpos[1]) @ self.ttmat_fn(f3_medial_t)
        mf_t_distal = mf_t_medial @ self.rytmat_fn(mf_qpos[2]) @ self.ttmat_fn(f3_distal_t)
        mf_t_ftp = mf_t_distal @ self.rytmat_fn(mf_qpos[3]) @ self.ttmat_fn(f3_tip_t)
        mf_jacp = cs.jacobian(mf_t_ftp[0:3, -1], q_pos)
        self.mf_J_fn = cs.Function('mf_J_fn', [mf_input_param], [mf_jacp, mf_t_ftp[0:3, -1]])

        # ring finger
        rf_input_param = cs.vvcat([rf_qpos, f3_t_param])
        rf_t_base = t_palm @ self.ttmat_fn([0, -0.0435, -0.001542]) @ self.quattmat_fn([0.999048, 0.0436194, 0, 0])
        rf_t_proximal = rf_t_base @ self.rztmat_fn(rf_qpos[0]) @ self.ttmat_fn(f3_proximal_t)
        rf_t_medial = rf_t_proximal @ self.rytmat_fn(rf_qpos[1]) @ self.ttmat_fn(f3_medial_t)
        rf_t_distal = rf_t_medial @ self.rytmat_fn(rf_qpos[2]) @ self.ttmat_fn(f3_distal_t)
        rf_t_ftp = rf_t_distal @ self.rytmat_fn(rf_qpos[3]) @ self.ttmat_fn(f3_tip_t)
        rf_jacp = cs.jacobian(rf_t_ftp[0:3, -1], q_pos)
        self.rf_J_fn = cs.Function('rf_J_fn', [rf_input_param], [rf_jacp, rf_t_ftp[0:3, -1]])

        # Thumb
        th_input_param = cs.vvcat([th_qpos, th_t_param])
        th_t_base = t_palm @ self.ttmat_fn([-0.0182, 0.019333, -0.045987]) @ self.quattmat_fn([0.477714, -0.521334, -0.521334, -0.477714])
        th_t_proximal = th_t_base @ self.rxtmat_fn(-th_qpos[0]) @ self.ttmat_fn(th_proximal_t)
        th_t_medial = th_t_proximal @ self.rztmat_fn(th_qpos[1]) @ self.ttmat_fn(th_medial_t)
        th_t_distal = th_t_medial @ self.rytmat_fn(th_qpos[2]) @ self.ttmat_fn(th_distal_t)
        th_t_ftp = th_t_distal @ self.rytmat_fn(th_qpos[3]) @ self.ttmat_fn(th_tip_t)
        th_jacp = cs.jacobian(th_t_ftp[0:3, -1], q_pos)
        self.th_J_fn = cs.Function('thtp_pos_fd_fn', [th_input_param], [th_jacp, th_t_ftp[0:3, -1]])

        self.fn_list = []
        self.fn_list.append(self.ff_J_fn)
        self.fn_list.append(self.mf_J_fn)
        self.fn_list.append(self.rf_J_fn)
        self.fn_list.append(self.th_J_fn)
        
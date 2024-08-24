import mujoco
import time

import casadi as cs
import numpy as np
np.set_printoptions(suppress=True)

from dynamics.allegro_dynamics import allegro_dynamics

from utils import rotations
from utils import pointcloudMap

class Contact:
    def __init__(self, param):
        self.param_ = param

        self.hand_dyn = allegro_dynamics()
        self.lseContactInit()

    def getQueryParam(self, query_pt_dict):
        finger_idx = query_pt_dict['finger_idx']
        joint_status = query_pt_dict['joint_status']
        joint_position = query_pt_dict['joint_position']
        body_translation = query_pt_dict['body_translation']

        if finger_idx != 3:
            linkt_list = []
            linkt_list.append(np.array([0, 0, 0.0164]))
            linkt_list.append(np.array([0, 0, 0.054]))
            linkt_list.append(np.array([0, 0, 0.0384]))
            linkt_list.append(np.array([0, 0, 0.0267]))
        elif finger_idx == 3:
            linkt_list = []
            linkt_list.append(np.array([-0.027, 0.005, 0.0399]))
            linkt_list.append(np.array([0, 0, 0.0177]))
            linkt_list.append(np.array([0, 0, 0.0514]))
            linkt_list.append(np.array([0, 0, 0.0423]))

        body_idx = sum(joint_status) - 1

        q_states = joint_position * joint_status
        param_t = np.concatenate((
            joint_status[0] * linkt_list[0],
            joint_status[1] * linkt_list[1],
            joint_status[2] * linkt_list[2],
            joint_status[3] * linkt_list[3]
        ))

        if body_idx != -1:
            param_t[body_idx*3:body_idx*3+3] = body_translation

        cs_param = np.concatenate((q_states, param_t))

        return cs_param

    def jacCalculate(self, joint_states):

        query_finger_body_pts = self.initContactQueryPoints(joint_states)


        jac_data_prep_time = time.time()
        fingers_batch_list = [[],[],[],[]]
        fingers_info_list = [[],[],[],[]]

        for point_i in range(len(query_finger_body_pts)):
            finger_idx = query_finger_body_pts[point_i]['finger_idx']
            body_idx = sum(query_finger_body_pts[point_i]['joint_status']) - 1
            param = self.getQueryParam(query_finger_body_pts[point_i])

            fingers_batch_list[finger_idx].append(param)
            fingers_info_list[finger_idx].append(np.array([finger_idx, body_idx]))
        
        for finger_i in range(4):
            fingers_batch_list[finger_i] = np.array(fingers_batch_list[finger_i]).T
            fingers_info_list[finger_i] = np.array(fingers_info_list[finger_i])
        print('jac_data_prep_time = ', 1000 * (time.time() - jac_data_prep_time))

        query_points = []
        jac_list = []
        info_list = []
        for finger_i in range(4):
            Jac_one_finger, fk_point = self.hand_dyn.fn_list[finger_i](fingers_batch_list[finger_i])

            # mind the reshape order 'F' 'C'
            jac_tmp = Jac_one_finger.toarray().reshape((3,16,-1), order='F')
            
            # concatenate all joints Jacobian
            Jac = np.zeros((3, self.param_.n_qvel_, jac_tmp.shape[2]))
            Jac[:,6:,:] = jac_tmp

            query_points.append(fk_point)
            jac_list.append(Jac)
            info_list.append(fingers_info_list[finger_i])

        query_points = np.concatenate(query_points, axis=-1).T
        jac_list = np.concatenate(jac_list, axis=-1).transpose((2,0,1))
        info_list = np.concatenate(info_list, axis=0)

        return query_points, jac_list, info_list

        
    def detectOnce(self, curr_x, curr_q, simulator):
        obj_pos = curr_q[0:3]
        obj_quat = curr_q[3:7]
        joint_states = curr_q

        # -------------------------------
        #        obj body frame
        # -------------------------------
        obj_rotmat = rotations.quat2rotmat(obj_quat)
        body_to_world_matrix = np.eye(4)
        body_to_world_matrix[:3, :3] = obj_rotmat
        body_to_world_matrix[:3, 3] = obj_pos

        # -------------------------------
        #     ground points process
        # -------------------------------
        # ground contact query points in world frame
        gd_filtered_query_points, gd_filtered_collision_points, gd_filtered_sdf_phi_list, gd_filtered_normal_list = self.groundProcess(
            obj_pos, body_to_world_matrix)

        # -------------------------------
        #     ftps points process
        # -------------------------------
        jac_time = time.time()
        ftp_query_pts_world, jac_list, info_list = self.jacCalculate(joint_states)
        print('jac_time = ', 1000 * (time.time() - jac_time))

        ftp_points_world, ftp_collision_points_list, ftp_normal_list, ftp_sdf_phi_list, ft_filtered_jac = self.fingerProcess(
            ftp_query_pts_world, jac_list, info_list, body_to_world_matrix)

        
        # -------------------------------
        #    concatenate all contacts
        # -------------------------------
        if ftp_points_world.shape[0] != 0 and gd_filtered_query_points.shape[0] != 0:
            query_points = np.vstack((ftp_points_world, gd_filtered_query_points))
            collision_points = np.vstack((ftp_collision_points_list, gd_filtered_collision_points))
            sdf_phi_list = np.concatenate((ftp_sdf_phi_list, gd_filtered_sdf_phi_list))
            normal_list = np.vstack((ftp_normal_list, gd_filtered_normal_list))
        elif ftp_points_world.shape[0] != 0:
            query_points = ftp_points_world
            collision_points = ftp_collision_points_list
            sdf_phi_list = ftp_sdf_phi_list
            normal_list = ftp_normal_list
        elif gd_filtered_query_points.shape[0] != 0:
            query_points = gd_filtered_query_points
            collision_points = gd_filtered_collision_points
            sdf_phi_list = gd_filtered_sdf_phi_list
            normal_list = gd_filtered_normal_list
        else:
            query_points = np.array([])
            collision_points = np.array([])
            sdf_phi_list = np.array([])
            normal_list = np.array([])
            
        phi_vec, jac_n, jac_f = self.contactInfoCalculate(obj_pos, obj_quat, query_points, collision_points, normal_list, sdf_phi_list, ft_filtered_jac)

        return phi_vec, jac_n, jac_f
    
    def contactInfoCalculate(self, obj_pos, obj_quat, query_points, collision_points, normal_list, sdf_phi_list, ft_filtered_jac):
        
        con_phi_list = []
        # con_jac_list = []
        con_jac_n_list = []
        con_jac_f_list = []
        
        for i in range(sdf_phi_list.shape[0]):
            con_pos = 0.5 * (collision_points[i] + query_points[i])
            con_phi = sdf_phi_list[i] * 0.5
            con_normal = -normal_list[i]
            
            if i < ft_filtered_jac.shape[0]:
                con_frame = self.calcContactFrame(contact_normal=con_normal)
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))
                
                jacp1 = ft_filtered_jac[i]
                
                jacp2 = np.zeros((3, self.param_.n_qvel_))
                jacp2[:, 0:3] = np.eye(3)
                jacp2[:, 3:6] = -rotations.skew(con_pos - obj_pos) @ rotations.quat2rotmat(obj_quat)

                jacp = jacp2 - jacp1
                
                con_jacp = con_frame_pmd.T @ jacp
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                
            else:
                jacp = np.zeros((3, self.param_.n_qvel_))
                jacp[:, 0:3] = np.eye(3)
                jacp[:, 3:6] = -rotations.skew(con_pos - obj_pos) @ rotations.quat2rotmat(obj_quat)

                con_frame = self.calcContactFrame(contact_normal=con_normal)
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                con_jacp = con_frame_pmd.T @ jacp
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                
            con_phi_list.append(con_phi)
            # con_jac_list.append(con_jac)
            con_jac_n_list.append(con_jacp_n)
            con_jac_f_list.append(con_jacp_f)
        
        phi_vec, jac_n, jac_f = self.qsimUpdateJacobian(dict(
            con_phi_list=con_phi_list,
            con_jac_n_list=con_jac_n_list,
            con_jac_f_list=con_jac_f_list))
        
        return phi_vec, jac_n, jac_f
    
    def calcContactFrame(self, contact_normal):

        con_tan_x = np.cross(contact_normal, self.param_.con_vec_)
        con_tan_y = np.cross(contact_normal, con_tan_x)

        con_tan_x = con_tan_x / (np.linalg.norm(con_tan_x) + 1e-15)
        con_tan_y = con_tan_y / (np.linalg.norm(con_tan_y) + 1e-15)
        con_normal = contact_normal / (np.linalg.norm(contact_normal) + 1e-15)

        return np.vstack((con_normal, con_tan_x, con_tan_y)).T
    
    def frameTrans(self, points, T):
        aug_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        points_transformed = (T @ aug_points.T).T[:,0:3]
        return points_transformed
    
    def groundProcess(self, obj_pos, body_to_world_matrix):

        gd_sdf_phi_list = []
        gd_normal_list = []
        gd_collision_points_list = []
        gd_query_points_world = self.param_.gd_points_ + np.append(obj_pos[0:2], 0.0)
        gd_query_points_body = self.frameTrans(gd_query_points_world, np.linalg.inv(body_to_world_matrix))
        for i in range(gd_query_points_body.shape[0]):
            # drop the ground query points that out of palm
            if gd_query_points_world[i,0] > 0:
                continue
            distance, normal, c_point = self.pointLSEContact(gd_query_points_body[i], body_to_world_matrix)
            gd_sdf_phi_list.append(distance)
            gd_normal_list.append(normal)
            gd_collision_points_list.append(c_point)
        gd_query_points_world = gd_query_points_world
        gd_collision_points_list = np.array(gd_collision_points_list)
        gd_sdf_phi_list = np.array(gd_sdf_phi_list)
        gd_normal_list = np.array(gd_normal_list)
        gd_filtered_query_points, gd_filtered_collision_points, gd_filtered_sdf_phi_list, gd_filtered_normal_list = self.gdFilter(
            gd_query_points_world, gd_collision_points_list, gd_sdf_phi_list, gd_normal_list, 1)

        return gd_filtered_query_points, gd_filtered_collision_points, gd_filtered_sdf_phi_list, gd_filtered_normal_list

    
    def fingerProcess(self, ftp_query_pts_world, jac_list, info_list, body_to_world_matrix):
        # -----------------------
        #         batch 
        # -----------------------
        ftp_query_pts_body = self.frameTrans(ftp_query_pts_world, np.linalg.inv(body_to_world_matrix))
        temp_ftp_sdf_phi_list, temp_ftp_normal_list, temp_ftp_collision_points_list \
            = self.pointcloudLSEContact(ftp_query_pts_body.T, body_to_world_matrix)
        temp_ftp_sampled_points_world = ftp_query_pts_world
        temp_ftp_jac_list = jac_list
        ft_filtered_query_points, \
        ft_filtered_collision_points, \
        ft_filtered_sdf_phi, ft_filtered_normal, \
        ft_filtered_jac \
            = self.fingerFilter(
            temp_ftp_sampled_points_world, \
            temp_ftp_collision_points_list, \
            temp_ftp_sdf_phi_list, \
            temp_ftp_normal_list, \
            temp_ftp_jac_list, \
            info_list, \
            15)
        
        ftp_points_world = np.squeeze(ft_filtered_query_points)
        ftp_collision_points_list = ft_filtered_collision_points
        ftp_normal_list = ft_filtered_normal
        ftp_sdf_phi_list = ft_filtered_sdf_phi
        
        return ftp_points_world, ftp_collision_points_list, ftp_normal_list, ftp_sdf_phi_list, ft_filtered_jac
    
    def pointLSEContact(self, point, T):
        casadi_sdf, casadi_sdf_gradient_body = self.contact_lse_fn_(point)
        casadi_collision_point_body = point - casadi_sdf * casadi_sdf_gradient_body
        aug_casadi_sdf_gradient_body = np.append(casadi_sdf_gradient_body, 1)
        aug_casadi_collision_point_body = np.append(casadi_collision_point_body, 1)
        casadi_sdf_gradient_world = (T @ aug_casadi_sdf_gradient_body)[0:3]
        casadi_collision_point_world = (T @ aug_casadi_collision_point_body)[0:3]
        
        return casadi_sdf.toarray()[0,0], casadi_sdf_gradient_world, casadi_collision_point_world
    
    def pointcloudLSEContact(self, pointcloud, T):
        casadi_sdf_batch, casadi_sdf_gradient_body_batch = self.contact_lse_fn_(pointcloud)
        casadi_collision_point_body_batch = pointcloud - casadi_sdf_gradient_body_batch.toarray() * casadi_sdf_batch.toarray()
        casadi_collision_point_world_batch = self.frameTrans(casadi_collision_point_body_batch.T, T)
        casadi_sdf_gradient_world_batch = self.frameTrans(casadi_sdf_gradient_body_batch.T, T)
        
        return np.squeeze(casadi_sdf_batch.T), casadi_sdf_gradient_world_batch, casadi_collision_point_world_batch
    
    def lseContactInit(self):
        self.pointcloud_tree_ = pointcloudMap.pointcloudMap(self.param_.pointcloud_path_)

        # data
        self.cur_point_mat_ = self.pointcloud_tree_.downsampled_points_
        self.cur_normal_mat_ = self.pointcloud_tree_.downsampled_normals_

        cur_point_mat_DM = cs.transpose(cs.DM(self.cur_point_mat_))
        cur_normal_mat_DM = cs.transpose(cs.DM(self.cur_normal_mat_))

        xq = cs.SX.sym('xq', 3)
        obj_to_xq = xq - cur_point_mat_DM
        dot_result = obj_to_xq * cur_normal_mat_DM
        distance = cs.sum1(dot_result) * self.param_.contact_lse_smoothness_delta_
        
        logsumexp_result = cs.logsumexp(distance.T)
        
        max_xq = logsumexp_result / self.param_.contact_lse_smoothness_delta_

        sdf_gradient = cs.gradient(max_xq, xq)

        self.contact_lse_fn_ = cs.Function('contact_lse_fn_', [xq], [max_xq, sdf_gradient])

    def gdFilter(self, query_points, collision_points, sdf_phi_list, normal_list, keep_point_number):
        filtered_indices = np.argsort(sdf_phi_list)[:keep_point_number]
        
        filtered_query_points = query_points[filtered_indices]
        filtered_collision_points = collision_points[filtered_indices]
        filtered_sdf_phi_list = sdf_phi_list[filtered_indices]
        filtered_normal_list = normal_list[filtered_indices]
        
        return filtered_query_points, filtered_collision_points, filtered_sdf_phi_list, filtered_normal_list
    
    def fingerFilter(self, query_points, collision_points, sdf_phi_list, normal_list, jac_list, info_list, keep_point_number):

        filtered_indices = []
        for finger_idx in range(4):
            for link_idx in range(4):
                link_indices = np.where((info_list[:, 0] == finger_idx) & (info_list[:, 1] == link_idx))[0]
                if link_indices.shape[0] == 0:
                    continue
                curr_link_phi_list = sdf_phi_list[link_indices]
                point_indices = np.argsort(curr_link_phi_list)[:1][0]
                filtered_indices.append(link_indices[point_indices])

        filtered_query_points = query_points[filtered_indices]
        filtered_collision_points = collision_points[filtered_indices]
        filtered_sdf_phi_list = sdf_phi_list[filtered_indices]
        filtered_normal_list = normal_list[filtered_indices]
        filtered_jac_list = jac_list[filtered_indices]

        filtered_indices = np.where(filtered_sdf_phi_list < self.param_.contact_margin_)
        filtered_query_points = filtered_query_points[filtered_indices]
        filtered_collision_points = filtered_collision_points[filtered_indices]
        filtered_sdf_phi_list = filtered_sdf_phi_list[filtered_indices]
        filtered_normal_list = filtered_normal_list[filtered_indices]
        filtered_jac_list = filtered_jac_list[filtered_indices]
        
        return filtered_query_points, filtered_collision_points, filtered_sdf_phi_list, filtered_normal_list, filtered_jac_list
    
    def pointTrans(self, point, T):
        aug_points = np.append(point, 1)
        point_transformed = (T @ aug_points.T).T[0:3]
        return point_transformed
    
    def qsimUpdateJacobian(self, contacts=None):
        # parse the input
        # con_jac_list = contacts['con_jac_list']
        con_phi_list = contacts['con_phi_list']
        con_jac_n_list = contacts['con_jac_n_list']
        con_jac_f_list = contacts['con_jac_f_list']

        print('contact size = ', len(con_phi_list))

        # fill the phi_vec
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))  # this is very,very important for soft sensitivity analysis
        jac_n = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        jac_f = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        for i in range(len(con_phi_list)):
            if i >= self.param_.max_ncon_:
                break
            phi_vec[4 * i: 4 * i + 4] = con_phi_list[i]
            jac_n[4 * i: 4 * i + 4] = con_jac_n_list[i]
            jac_f[4 * i: 4 * i + 4] = con_jac_f_list[i]

        return phi_vec, jac_n, jac_f
    
    def initContactQueryPoints(self, joint_states):

        test_query_pts = []

        ff_joint_states = joint_states[7:11]
        mf_joint_states = joint_states[11:15]
        rf_joint_states = joint_states[15:19]
        th_joint_states = joint_states[19:23]

        # --------------------------------------------------------------------------

        # fingertip of thumb finger
        sphere_radius = 0.012
        for theta in np.linspace(0, 2*np.pi, 10):
            for phi in np.linspace(0, np.pi/2, 5):
                # Convert spherical coordinates to Cartesian coordinates
                x = sphere_radius * np.sin(phi) * np.cos(theta)
                y = sphere_radius * np.sin(phi) * np.sin(theta)
                z = sphere_radius * np.cos(phi) + 0.012 + 0.0313

                th_query_pt = dict(
                    finger_idx = 3,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = th_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(th_query_pt)

        cylinder_radius = 0.012
        for alpha in np.linspace(0, 2*np.pi, 10):
            for height in np.linspace(0.0, 0.012 + 0.0313, 10):
                # Convert spherical coordinates to Cartesian coordinates
                x = cylinder_radius * np.cos(alpha)
                y = cylinder_radius * np.sin(alpha)
                z = height

                th_query_pt = dict(
                    finger_idx = 3,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = th_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(th_query_pt)

        # fingertips of first three finger
        sphere_radius = 0.012
        for theta in np.linspace(0, 2*np.pi, 10):
            for phi in np.linspace(0, np.pi/2, 5):
                # Convert spherical coordinates to Cartesian coordinates
                x = sphere_radius * np.sin(phi) * np.cos(theta)
                y = sphere_radius * np.sin(phi) * np.sin(theta)
                z = sphere_radius * np.cos(phi) + 0.012 + 0.0157

                ff_query_pt1 = dict(
                    finger_idx = 0,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = ff_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(ff_query_pt1)

                mf_query_pt1 = dict(
                    finger_idx = 1,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = mf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(mf_query_pt1)

                rf_query_pt1 = dict(
                    finger_idx = 2,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = rf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(rf_query_pt1)

        cylinder_radius = 0.014
        for alpha in np.linspace(0, 2*np.pi, 10):
            for height in np.linspace(0, 0.0384, 3):
                # Convert spherical coordinates to Cartesian coordinates
                x = cylinder_radius * np.cos(alpha)
                y = cylinder_radius * np.sin(alpha)
                z = height

                ff_query_pt1 = dict(
                    finger_idx = 0,
                    joint_status = np.array([1, 1, 1, 0]),
                    joint_position = ff_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(ff_query_pt1)

                mf_query_pt1 = dict(
                    finger_idx = 1,
                    joint_status = np.array([1, 1, 1, 0]),
                    joint_position = mf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(mf_query_pt1)

                rf_query_pt1 = dict(
                    finger_idx = 2,
                    joint_status = np.array([1, 1, 1, 0]),
                    joint_position = rf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(rf_query_pt1)

        # second joint of first three finger
        for width_sample in np.linspace(-0.01345, 0.01345, 2):
            for height_sample in np.linspace(0, 0.054, 5):
                ff_query_pt1 = dict(
                    finger_idx = 0,
                    joint_status = np.array([1, 1, 0, 0]),
                    joint_position = ff_joint_states,
                    body_translation = np.array([0.0098, width_sample, height_sample])
                )
                test_query_pts.append(ff_query_pt1)

                mf_query_pt1 = dict(
                    finger_idx = 1,
                    joint_status = np.array([1, 1, 0, 0]),
                    joint_position = mf_joint_states,
                    body_translation = np.array([0.0098, width_sample, height_sample])
                )
                test_query_pts.append(mf_query_pt1)

                rf_query_pt1 = dict(
                    finger_idx = 2,
                    joint_status = np.array([1, 1, 0, 0]),
                    joint_position = rf_joint_states,
                    body_translation = np.array([0.0098, width_sample, height_sample])
                )
                test_query_pts.append(rf_query_pt1)

        cylinder_radius = 0.012
        for alpha in np.linspace(0, 2*np.pi, 10):
            for height in np.linspace(0, 0.012, 3):
                # Convert spherical coordinates to Cartesian coordinates
                x = cylinder_radius * np.cos(alpha)
                y = cylinder_radius * np.sin(alpha)
                z = height + 0.0157

                ff_query_pt1 = dict(
                    finger_idx = 0,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = ff_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(ff_query_pt1)

                mf_query_pt1 = dict(
                    finger_idx = 1,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = mf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(mf_query_pt1)

                rf_query_pt1 = dict(
                    finger_idx = 2,
                    joint_status = np.array([1, 1, 1, 1]),
                    joint_position = rf_joint_states,
                    body_translation = np.array([x, y, z])
                )
                test_query_pts.append(rf_query_pt1)

        # thrid joint of thumb finger
        for width_sample in np.linspace(-0.01345, 0.01345, 3):
            for height_sample in np.linspace(0, 0.0514, 5):
                th_query_pt = dict(
                    finger_idx = 3,
                    joint_status = np.array([1, 1, 1, 0]),
                    joint_position = th_joint_states,
                    body_translation = np.array([0.0098, width_sample, height_sample])
                )
                test_query_pts.append(th_query_pt)

        return test_query_pts
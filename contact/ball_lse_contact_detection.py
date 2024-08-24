import casadi as cs
import numpy as np
np.set_printoptions(suppress=True)

from utils import rotations
from utils import pointcloudMap

from simulator.fts_simulator import MjSimulator

class Contact:
    def __init__(self, param):
        self.param_ = param
        
        if self.param_.contact_mode_ == 'lse':
            self.lseContactInit()
            
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

        self.contact_lse_fn_ = cs.Function('contact_lse_fn_', [xq], [max_xq])
        self.sdf_gradient_fn_ = cs.Function('sdf_gradient_fn_', [xq], [sdf_gradient])
        
    def detectOnce(self, curr_q, simulator:MjSimulator=None):
        # -------------------------------
        #        system states
        # -------------------------------
        obj_pos = curr_q[0:3]
        obj_quat = curr_q[3:7]
        ft1_pos = curr_q[7:10]
        ft2_pos = curr_q[10:13]
        ft3_pos = curr_q[13:16]
        
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
        gd_sdf_phi_list = []
        gd_normal_list = []
        gd_collision_points_list = []
        gd_query_points_world = self.param_.gd_points_ + np.append(obj_pos[0:2], 0.0)
        gd_query_points_body = self.frameTrans(gd_query_points_world, np.linalg.inv(body_to_world_matrix))
        for i in range(gd_query_points_body.shape[0]):
            distance, normal, c_point = self.pointLSEContact(gd_query_points_body[i], body_to_world_matrix)
            gd_sdf_phi_list.append(distance)
            gd_normal_list.append(normal)
            gd_collision_points_list.append(c_point)
        gd_query_points_world = gd_query_points_world
        gd_collision_points_list = np.array(gd_collision_points_list)
        gd_sdf_phi_list = np.array(gd_sdf_phi_list)
        gd_normal_list = np.array(gd_normal_list)
        gd_filtered_query_points, gd_filtered_collision_points, gd_filtered_sdf_phi_list, gd_filtered_normal_list = self.queryPointDistanceFilter(
            gd_query_points_world, gd_collision_points_list, gd_sdf_phi_list, gd_normal_list, 1)
        # if gd_filtered_sdf_phi_list[0] > 0.01:
        #     gd_filtered_query_points = np.array([])
        #     gd_filtered_collision_points = np.array([])
        #     gd_filtered_sdf_phi_list = np.array([])
        #     gd_filtered_normal_list = np.array([])
        
        # -------------------------------
        #     ftps points process
        # -------------------------------
        # default fingertips contact query points in world frame
        ftp_points_world = np.vstack((ft1_pos, ft2_pos, ft3_pos))
        
        ftp_sampled_points_world = []
        ftp_sdf_phi_list = []
        ftp_normal_list = []
        ftp_collision_points_list = []
        fts_query_index_list = []
        for i in range(ftp_points_world.shape[0]):
            ftp_points_ftbody = self.pointTrans(ftp_points_world[i], np.linalg.inv(body_to_world_matrix))
            distance, normal, c_point = self.pointLSEContact(ftp_points_ftbody, body_to_world_matrix)

            if distance > 0.015:
                continue
            ftp_sampled_points_world.append(ftp_points_world[i])
            ftp_sdf_phi_list.append(distance)
            ftp_normal_list.append(normal)
            ftp_collision_points_list.append(c_point)
            fts_query_index_list.append(i)
        
        fts_query_index = np.array(fts_query_index_list)
        ftp_points_world = np.squeeze(np.array(ftp_sampled_points_world)).reshape(-1,3)
        ftp_collision_points_list = np.squeeze(np.array(ftp_collision_points_list)).reshape(-1,3)
        ftp_normal_list = np.squeeze(np.array(ftp_normal_list)).reshape(-1,3)
        ftp_sdf_phi_list = np.squeeze(np.array(ftp_sdf_phi_list)).reshape(-1,)
        
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
        
        # prepare for return
        con_phi_list = []
        con_frame_list = []
        con_normal_list = []
        con_pos_list = []
        con_jac_list = []
        con_jac_n_list = []
        con_jac_f_list = []
        for i in range(collision_points.shape[0]):
            con_pos = 0.5 * (collision_points[i] + query_points[i])

            con_phi = sdf_phi_list[i] * 0.5

            con_normal = -normal_list[i]

            # compute the jacobian
            if i < fts_query_index.shape[0]:
                ftp_index = fts_query_index[i]
            # fts_query_points = np.vstack((ft1_pos, ft2_pos, ft3_pos))

                # 1:
                # derivative of the translation velocity of the fingertips_i (v_fti) 
                # with respect to the generlized velocity (vc, w_c, v_ft1, v_ft2, v_ft3)
                jacp1 = np.zeros((3, self.param_.n_qvel_))
                jacp1[:, 6 + ftp_index * 3:6 + ftp_index * 3 + 3] = np.eye(3)

                # 2:
                # derivative of the translation velocity of the cube (v_c)
                # with respect to the generlized velocity (vc, w_c, v_ft1, v_ft2, v_ft3)
                jacp2 = np.zeros((3, self.param_.n_qvel_))
                jacp2[:, 0:3] = np.eye(3)
                jacp2[:, 3:6] = -rotations.skew(con_pos - obj_pos) @ rotations.quat2rotmat(obj_quat)
                # the Jacobian of the relative velocity of the contact pair
                jacp = jacp2 - jacp1

                con_frame = self.calcContactFrame(contact_normal=con_normal)
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                con_jacp = con_frame_pmd.T @ jacp
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                # con_jac = con_jacp_n + self.param_.mu_fingertip_ * con_jacp_f
                # con_jac_list.append(con_jac)
                con_jac_n_list.append(con_jacp_n)
                con_jac_f_list.append(con_jacp_f)

            else:
                # con_normal = np.array([0, 0, 1.0])
                jacp = np.zeros((3, self.param_.n_qvel_))
                jacp[:, 0:3] = np.eye(3)
                jacp[:, 3:6] = -rotations.skew(con_pos - obj_pos) @ rotations.quat2rotmat(obj_quat)

                con_frame = self.calcContactFrame(contact_normal=con_normal)
                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                con_jacp = con_frame_pmd.T @ jacp
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                # con_jac = con_jacp_n + self.param_.mu_table_ * con_jacp_f
                # con_jac_list.append(con_jac)
                con_jac_n_list.append(con_jacp_n)
                con_jac_f_list.append(con_jacp_f)

            con_pos_list.append(con_pos)
            con_phi_list.append(con_phi)
            con_normal_list.append(con_normal)
            con_frame_list.append(con_frame)
            
        phi_vec, jac_n_mat, jac_f_mat = self.qsimUpdateJacobian(
            dict(
            con_pos_list=con_pos_list,
            con_phi_list=con_phi_list,
            con_normal_list=con_normal_list,
            con_frame_list=con_frame_list,
            con_jac_n_list=con_jac_n_list,
            con_jac_f_list=con_jac_f_list
            )
        )
        
        return phi_vec, jac_n_mat, jac_f_mat
    
    def queryPointDistanceFilter(self, query_points, collision_points, sdf_phi_list, normal_list, keep_point_number):
        filtered_indices = np.argsort(sdf_phi_list)[:keep_point_number]
        
        filtered_query_points = query_points[filtered_indices]
        filtered_collision_points = collision_points[filtered_indices]
        filtered_sdf_phi_list = sdf_phi_list[filtered_indices]
        filtered_normal_list = normal_list[filtered_indices]
        
        return filtered_query_points, filtered_collision_points, filtered_sdf_phi_list, filtered_normal_list
    
    def pointLSEContact(self, point, T):
        casadi_sdf = self.contact_lse_fn_(point)
        casadi_sdf_gradient_body = self.sdf_gradient_fn_(point)
        casadi_collision_point_body = point - casadi_sdf * casadi_sdf_gradient_body
        aug_casadi_sdf_gradient_body = np.append(casadi_sdf_gradient_body, 1)
        aug_casadi_collision_point_body = np.append(casadi_collision_point_body, 1)
        casadi_sdf_gradient_world = (T @ aug_casadi_sdf_gradient_body)[0:3]
        casadi_collision_point_world = (T @ aug_casadi_collision_point_body)[0:3]
        
        return casadi_sdf.toarray()[0,0], casadi_sdf_gradient_world, casadi_collision_point_world
    
    def frameTrans(self, points, T):
        aug_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        points_transformed = (T @ aug_points.T).T[:,0:3]
        return points_transformed
    
    def pointTrans(self, point, T):
        aug_points = np.append(point, 1)
        point_transformed = (T @ aug_points.T).T[0:3]
        return point_transformed
        
    def calcContactFrame(self, contact_normal):

        con_tan_x = np.cross(contact_normal, self.param_.con_vec_)
        con_tan_y = np.cross(contact_normal, con_tan_x)

        con_tan_x = con_tan_x / (np.linalg.norm(con_tan_x) + 1e-15)
        con_tan_y = con_tan_y / (np.linalg.norm(con_tan_y) + 1e-15)
        con_normal = contact_normal / (np.linalg.norm(contact_normal) + 1e-15)

        return np.vstack((con_normal, con_tan_x, con_tan_y)).T
    
    def qsimUpdateJacobian(self, contacts=None):
        # parse the input
        # con_jac_list = contacts['con_jac_list']
        con_jac_n_list = contacts['con_jac_n_list']
        con_jac_f_list = contacts['con_jac_f_list']
        con_phi_list = contacts['con_phi_list']

        # fill the phi_vec
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))  # this is very,very important for soft sensitivity analysis
        # jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        jac_n_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        jac_f_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_qvel_))
        for i in range(len(con_phi_list)):
            phi_vec[4 * i: 4 * i + 4] = con_phi_list[i]
            # jac_mat[4 * i: 4 * i + 4] = con_jac_list[i]
            jac_n_mat[4 * i: 4 * i + 4] = con_jac_n_list[i]
            jac_f_mat[4 * i: 4 * i + 4] = con_jac_f_list[i]

        return phi_vec, jac_n_mat, jac_f_mat
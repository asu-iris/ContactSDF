import numpy as np
from scipy.spatial import cKDTree

class pointcloudMap(object):
    def __init__(self, pointcloud_path):
        # pointcloud input: .txt file with 6 columns: x, y, z, normal_x, normal_y, normal_z
        self.pointcloud_data_ = np.loadtxt(pointcloud_path, delimiter=',', dtype=np.float64)
        self.pointcloud_point_ = self.pointcloud_data_[:, :3]
        self.pointcloud_tree_ = cKDTree(self.pointcloud_point_)
        self.pointcloud_normal_ = self.pointcloud_data_[:, 3:6]
        
        subsampled_data_path = pointcloud_path[0:-4] + "_subsampled.txt"
        self.pointcloud_subsampled_data_ = np.loadtxt(subsampled_data_path, delimiter=',', dtype=np.float64)
        self.downsampled_points_ = self.pointcloud_subsampled_data_[:, :3]
        self.downsampled_normals_ = self.pointcloud_subsampled_data_[:, 3:6]
        print(self.downsampled_points_.shape)
        print(self.downsampled_normals_.shape)
        
    def nearestSearch(self, pos):
        dist, ind = self.pointcloud_tree_.query(pos)
        return dist, ind
    
    def getPoint(self, idx):
        return self.pointcloud_data_[idx][:3]
    
    def getPointNormal(self, idx):
        return self.pointcloud_data_[idx][3:6]
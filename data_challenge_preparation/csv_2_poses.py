"""
This file reformat the 6D pose CSV (x, y, z, yaw, pitch, roll) into the "CrossLoc Benchmark Datasets Setup" format.
https://github.com/TOPO-EPFL/CrossLoc-Benchmark-Datasets
CSV input output format :
x, y, z, yaw, pitch, roll
Output format : 4x4 homogeneous extrinsic camera matrix.
---
regis.longchamp@epfl.ch
"""

import os
import numpy as np

from config import settings
from utils_angles import wgs84_to_ecef,sixd_array_2_pose
from os import listdir
from os.path import isfile, join



def sixd_csv_array_to_txt_pose():
  """
  Read the 6D position from the CSV file and transform them into numpy array. 
  Save the numpy arry into a text file
  :return: None
  """
  path_csv_file = settings.path_file_poses_csv
  origin_xyz = np.array(wgs84_to_ecef(settings.origin_of_local_coordinate_system_y_wgs84,
                                      settings.origin_of_local_coordinate_system_x_wgs84,
                                      settings.origin_of_local_coordinate_system_z_wgs84))

  poses = np.genfromtxt(path_csv_file, delimiter=',')

  pose_files = [f for f in listdir(settings.path_folder_poses_txt) if isfile(join(settings.path_folder_poses_txt, f))]

  for i,pose in enumerate(poses):
    lng, lat, alt, azimuth, tilt, roll = list(pose)
    matrix_angles = sixd_array_2_pose([lng, lat, alt, azimuth, tilt, roll])
    global_coordinates = np.array(wgs84_to_ecef(lat, lng, alt)) # erreur
    local_coordinates = global_coordinates - origin_xyz
    r = np.concatenate((matrix_angles, np.array([local_coordinates]).transpose()), axis=1)
    r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)

    file_name = pose_files[i]
    folder_path = settings.path_folder_poses_txt.replace('poses','poses_verification')
    if not os.path.exists(folder_path) :
      os.makedirs(folder_path)
    file_path = os.path.join(folder_path,file_name)
    np.savetxt(file_path,r)



if __name__ == '__main__':
    sixd_csv_array_to_txt_pose()

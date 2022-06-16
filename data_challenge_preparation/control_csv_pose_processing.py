"""
This script aim to verify the transfmation pose_2_csv and csv_2_pose
it compares the inputs of pose_2_csv with csv_2_pose outputs
---
Usage : Run the script, paths are based on the config files
---
regis.longchamp@epfl.ch
"""

import os
from os import listdir
from os.path import isfile, join
import numpy as np

from config import settings

def contol_process_csv_pose_transformation() :
    folder_path_poses_init = settings.path_folder_poses_txt 
    poses_init = [os.path.join(folder_path_poses_init,f) for f in listdir(folder_path_poses_init) if isfile(join(folder_path_poses_init, f))]

    folder_path = settings.path_folder_poses_txt.replace('poses','poses_verification')
    poses_cal = [os.path.join(folder_path,f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    list_max_diff = []

    for i,path_pose_init in enumerate(poses_init) :
        pose_init = np.loadtxt(path_pose_init)
        pose_cal = np.loadtxt(poses_cal[i])
        subtract = np.subtract(pose_init,pose_cal)
        max_diff = max(subtract.min(), subtract.max(), key=abs)
        list_max_diff.append(max_diff)

    overall_max_value = max(list_max_diff)
    print(overall_max_value)

if __name__ == '__main__':
    contol_process_csv_pose_transformation()

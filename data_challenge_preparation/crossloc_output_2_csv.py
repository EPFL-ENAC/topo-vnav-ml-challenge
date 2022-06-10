"""
This file reformat the outputs of the "CrossLoc localization" tool into CSV.
https://github.com/TOPO-EPFL/CrossLoc
CSV row output format :
x, y, z, yaw, pitch, roll
---
regis.longchamp@epfl.ch
"""

import os
import numpy as np
import zipfile
import shutil
import csv


from config import settings
from utils_angles import extract_angles_from_pose



def numpy_loader(path: str) -> np.ndarray:
    """
    Load nupy file into arrays
    """
    if os.path.exists(path):
        nb_data = np.load(path)
    else:
        raise Exception("File not found")
    return nb_data



def crossloc_output_2_list_poses_paths(file_name: str = 'pose_pred.npy') -> list:
    """
    Unzip crossloc results and return the list of npy files
    :param file_name: npy file name to search for, typically "pose_gt.npy" or "pose_pred.npy"
    :return: list of paths
    """
    result_folder = settings.crossloc_pose_result_folder

    list_of_folder = os.listdir(result_folder)
    list_of_npy_paths = []
    for item in list_of_folder:
        if item.endswith('.npz'):
            path_zip = os.path.join(result_folder, item)
            path_folder_output = os.path.join(result_folder, item.replace('.npz', ''))
            if os.path.exists(path_folder_output):
                shutil.rmtree(path_folder_output)
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(path_folder_output)
            path_npy = os.path.join(path_folder_output, file_name)
            if os.path.exists(path_npy):
                list_of_npy_paths.append(path_npy)
    return list_of_npy_paths




def pose_2_cvs_arrays(path_poses: [str, list], path_output_file : str):
    """
    Convert positions and angles from the pose into WGS84 and ° and write a result csv
    :param path_poses: path or list of paths of the npy pose files
    :return: none
    """

    row_csv_file = []

    if isinstance(path_poses, str):
        list_path_pose = [path_poses]
    else:
        list_path_pose = path_poses


    for i, npy_path_file in enumerate(list_path_pose):
        npy_data = numpy_loader(npy_path_file)
        coordinates = npy_data[0:3, 3]
        # row_csv_file.append(list(coordinates))
        lat, lng, alt, x, y, z, azimuth, tilt, roll ,x_local, y_local, z_local = extract_angles_from_pose(npy_data)

        current_row = [lat, lng, alt, azimuth, tilt, roll]

        row_csv_file.append(current_row)

    with open(path_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(row_csv_file)




def format_crossloc_prediction_output() :
    """
    Run the full translation from Crossloc results files to a CSV file that can be loaded in the codalab competition
    :param pose_file_name: pose file name, typically pose_pred.npy or pose_gt.npy
    :return: None
    """

    if settings.pose_pred_file_name and settings.path_file_out_pred :
        # Unzip Crossloc results and get the list of all *.npy
        list_of_npy_paths = crossloc_output_2_list_poses_paths(settings.pose_pred_file_name)

        # Create CSV based on the Crossloc results (previously unzip)
        pose_2_cvs_arrays(list_of_npy_paths, settings.path_file_out_pred)

    if settings.pose_gt_file_name and settings.path_file_out_gt :
        # Unzip Crossloc results and get the list of all *.npy
        list_of_npy_paths = crossloc_output_2_list_poses_paths(settings.pose_gt_file_name)

        # Create CSV based on the Crossloc results (previously unzip)
        pose_2_cvs_arrays(list_of_npy_paths, settings.path_file_out_gt)




if __name__ == '__main__':
    format_crossloc_prediction_output()



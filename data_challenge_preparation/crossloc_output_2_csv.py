import os
import numpy as np
import zipfile
import shutil
import csv
import pyproj
import argparse

from matrix_operation import get_rotation_ned_in_ecef, rotationMatrixToEulerAngles


def numpy_loader(path: str) -> np.ndarray:
    """
    Load nupy file into arrays
    """
    if os.path.exists(path):
        nb_data = np.load(path)
    else:
        raise Exception("File not found")
    return nb_data


def extract_angles_from_pose(pose, origin_of_local_coordinate_system_x: float = 0,
                             origin_of_local_coordinate_system_y: float = 0,
                             origin_of_local_coordinate_system_z: float = 0) -> list:
    """
    Extract the angle information from the pose array
    return: x,y,z ,azimuth, tilt, roll
    """

    # Pose ECEF local coordinates
    x_local = pose[0, 3]
    y_local = pose[1, 3]
    z_local = pose[2, 3]

    # Transform the center of the local coordinate system from WGS84 to ECEF
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(
        origin_of_local_coordinate_system_x,
        origin_of_local_coordinate_system_y,
        origin_of_local_coordinate_system_z)

    # Pose ECEF global coordinates
    x = ori_x + x_local
    y = ori_y + y_local
    z = ori_z + z_local

    # Pose WGS84 coordinates
    lat, lng, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)

    # Rotation matrix
    rot_ned_in_ecef = get_rotation_ned_in_ecef(lng, lat)

    # Remove the last line
    pose = pose[:3]

    # Remove the last column
    pose = pose[:, 0:3]

    # Insert column order
    pose = pose[0:3, [2, 0, 1]]

    # Get the multiplication matrix
    mat_rot_pose = np.matmul(np.linalg.inv(rot_ned_in_ecef), pose)

    # Get the angles
    roll, tilt, azimuth = rotationMatrixToEulerAngles(mat_rot_pose)

    return lat, lng, alt, x, y, z, azimuth, tilt, roll


def crossloc_res_unzip_get_npy_paths(result_folder: str, file_name: str = 'pose_pred.npy') -> list:
    """
    Unzip crossloc results and return the list of npy files
    :param result_folder: path of the folder that contains the crossloc result
    :param file_name: npy file name to search for, typically "pose_gt.npy" or "pose_pred.npy"
    :return: list of paths
    """
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


def numpy_array_2_csv(path_poses: [str, list], path_out_csv: str,
                                  origin_of_local_coordinate_system_x: float = 0,
                                  origin_of_local_coordinate_system_y: float = 0,
                                  origin_of_local_coordinate_system_z: float = 0):
    """
    Convert positions and angles from the pose into WGS84 and ° and write a result csv
    :param path_poses: path or list of paths of the npy pose files
    :param path_out_csv: output path
    :param origin_of_local_coordinate_system_x: local coordinate x
    :param origin_of_local_coordinate_system_y: local coordinate y
    :param origin_of_local_coordinate_system_z: local coordinate z
    :return: none
    """

    row_csv_file = []

    if isinstance(path_poses, str):
        list_path_pose = [path_poses]
    else:
        list_path_pose = path_poses

    for npy_path_file in list_path_pose:
        npy_data = numpy_loader(npy_path_file)
        coordinates = npy_data[0:3, 3]
        # row_csv_file.append(list(coordinates))
        lat, lng, alt, x, y, z, azimuth, tilt, roll = extract_angles_from_pose(npy_data,
                                                                               origin_of_local_coordinate_system_x,
                                                                               origin_of_local_coordinate_system_y,
                                                                               origin_of_local_coordinate_system_z)
        current_row = [lat, lng, alt, azimuth, tilt, roll]

        row_csv_file.append(current_row)

    with open(path_out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(row_csv_file)




def create_csv_for_codalab_submission(crossloc_pose_result_folder : str, pose_file_name : str,  path_file_csv : str,
                                      origin_of_local_coordinate_system_x : float,
                                      origin_of_local_coordinate_system_y : float,
                                      origin_of_local_coordinate_system_z : float) :
    """
    Run the full translation from Crossloc results files to a CSV file that can be loaded in the codalab competition
    :param crossloc_pose_result_folder: Path of the folder that contain the Crossloc result (with --save_pred).
    :param pose_file_name: pose file name, typically pose_pred.npy or pose_gt.npy
    :param path_file_csv: Path of the resulting CSV file
    :param origin_of_local_coordinate_system_x: X coordinate of the center of the local coordinate system
    :param origin_of_local_coordinate_system_y: Y coordinate of the center of the local coordinate system
    :param origin_of_local_coordinate_system_z: Z X coordinate of the center of the local coordinate system
    :return: None
    """

    # Unzip Crossloc results and get the list of all *.npy
    list_of_npy_paths = crossloc_res_unzip_get_npy_paths(crossloc_pose_result_folder, pose_file_name)

    # Create prediction CSV based on the Crossloc results (previously unzip)
    numpy_array_2_csv(list_of_npy_paths, path_file_csv, origin_of_local_coordinate_system_x,
                      origin_of_local_coordinate_system_y, origin_of_local_coordinate_system_z)


def _config_parser():
    parser = argparse.ArgumentParser(
        description='Translate Crossloc results files into a CSV file that can be loaded in the codalab competition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_folder', type=str, default=None,
                        help='Path of the folder that contain the Crossloc result (with --save_pred)')
    parser.add_argument('--in_files_name', type=str, default=None,
                        help='Pose file name, typically pose_pred.npy or pose_gt.npy')
    parser.add_argument('--out_csv', type=str, default=None,
                        help='Path of the resulting CSV file')
    parser.add_argument('--x', type=str, default=None,
                        help='X coordinate (WGS84) of the center of the local coordinate system')
    parser.add_argument('--y', type=str, default=None,
                            help='Y coordinate (WGS84) of the center of the local coordinate system')
    parser.add_argument('--z', type=str, default=None,
                            help='Z coordinate (WGS84) of the center of the local coordinate system')

    opt = parser.parse_args()

    return opt



def main() :
    opt = _config_parser()
    create_csv_for_codalab_submission(opt.in_folder,opt.in_files_name,opt.x,opt.y,opt.z)




if __name__ == '__main__':
    # main()

    # crossloc_output_2_csv.py --in_folder C:\projects\vnav\CrossLoc\weight\coord_pred_ckpt_iter_crossloc_urbanscape_in-place.net_zero --in_files_name pose_pred.npy --out_csv C:\projects\vnav\CrossLoc\weight\ok.csv --x 6.5668 --y 46.5191 --z 390


    origin_of_local_coordinate_system_x = 6.5668
    origin_of_local_coordinate_system_y = 46.5191
    origin_of_local_coordinate_system_z = 390
    crossloc_pose_result_folder = 'C:\\projects\\vnav\\CrossLoc\\weight\\coord_pred_ckpt_iter_crossloc_naturescape_in-place.net_test_drone_real'
    path_file_csv = 'C:\\projects\\vnav\\CrossLoc\\weight\\gt.csv'

    create_csv_for_codalab_submission(crossloc_pose_result_folder, 'pose_gt.npy', path_file_csv,
                                      origin_of_local_coordinate_system_x,
                                      origin_of_local_coordinate_system_y,
                                      origin_of_local_coordinate_system_z)


    # # Run crossloc with the parameter --save_pred to get all the poses.
    #
    #
    #
    #
    #
    #
    #
    #
    # # Unzip Crossloc results and get the list of all pose_gt.npy
    # list_of_npy_paths = crossloc_res_unzip_get_npy_paths(crossloc_pose_result_folder, 'pose_gt.npy')
    #
    # # Create ground truth CSV based on the Crossloc results (previously unzip)
    # ai_competition_result_file = 'C:\\projects\\vnav\\CrossLoc\\weight\\gt.csv'
    # numpy_array_2_csv(list_of_npy_paths, ai_competition_result_file,
    #                               origin_of_local_coordinate_system_x, origin_of_local_coordinate_system_y,
    #                               origin_of_local_coordinate_system_z)

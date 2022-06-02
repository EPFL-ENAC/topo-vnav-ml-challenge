import os
import csv
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
import cv2



def get_rotation_ned_in_ecef(lon, lat):
 """
 @param: lon, lat Longitude and latitude in degree
 @return: 3x3 rotation matrix of heading-pith-roll NED in ECEF coordinate system
 Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
 Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
 """
 # describe NED in ECEF
 lon = lon * np.pi / 180.0
 lat = lat * np.pi / 180.0
 # manual computation
 R_N0 = np.array([[np.cos(lon), -np.sin(lon), 0],
                  [np.sin(lon), np.cos(lon), 0],
                  [0, 0, 1]])
 R__E1 = np.array([[np.cos(-lat - np.pi / 2), 0, np.sin(-lat - np.pi / 2)],
                   [0, 1, 0],
                   [-np.sin(-lat - np.pi / 2), 0, np.cos(-lat - np.pi / 2)]])
 NED = np.matmul(R_N0, R__E1)
 assert abs(np.linalg.det(
  NED) - 1.0) < 1e-6, 'NED in NCEF rotation mat. does not have unit determinant, it is: {:.2f}'.format(
  np.linalg.det(NED))
 return NED


def ecef_to_geographic(x, y, z):
 # careful here lat,lon
 lat, lon, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)
 return [lon, lat, alt]



def get_pose_mat_original(pose):
 """
 Get 4x4 homogeneous matrix from Cesium-defined pose
 @input: cesium_pose 6d ndarray, [lat, lon, h, yaw, pitch, roll]
 lat, lon, h are in ECEF coordinate system
 yaw, pitch, roll are in degress
 @output: 4x4 homogeneous extrinsic camera matrix
 """
 x, y, z, yaw, pitch, roll = pose  # no need to do local conversion when in ECEF

 lon, lat, alt = ecef_to_geographic(x, y, z)
 rot_ned_in_ecef = get_rotation_ned_in_ecef(lon, lat)
 rot_pose_in_ned = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
 r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
 # transform coordinate system from NED to standard camera sys.
 r = r[0:3, [1, 2, 0]]
 r = np.concatenate((r, np.array([[x, y, z]]).transpose()), axis=1)
 r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
 return r


def get_pose_mat_original_angles(pose):
 """
 Get 4x4 homogeneous matrix from Cesium-defined pose
 @input: cesium_pose 6d ndarray, [lat, lon, h, yaw, pitch, roll]
 lat, lon, h are in ECEF coordinate system
 yaw, pitch, roll are in degress
 @output: 4x4 homogeneous extrinsic camera matrix
 """
 lat, lng, alt, yaw, pitch, roll = pose  # no need to do local conversion when in ECEF

 rot_ned_in_ecef = get_rotation_ned_in_ecef(lng, lat)
 rot_pose_in_ned = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
 r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
 # transform coordinate system from NED to standard camera sys.
 r = r[0:3, [1, 2, 0]]
 return r

def numpy_loader(path: str) -> np.ndarray:
    if os.path.exists(path):
        nb_data = np.load(path)
    else:
        raise Exception("File not found")
    return nb_data

def wgs84_to_ecef(lng, lat, alt):
 # careful here lat,lon

 x,y,z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(lng, lat, alt)
 return [x,y,z]



def csc_to_numpy(path_csv_file : str, origin_of_local_coordinate_system_x: float = 0,
                                  origin_of_local_coordinate_system_y: float = 0,
                                  origin_of_local_coordinate_system_z: float = 0):

    """
    Read pose from the CSV file and transform them into numpy array
    :param path_csv_file:
    :param origin_of_local_coordinate_system_x:
    :param origin_of_local_coordinate_system_y:
    :param origin_of_local_coordinate_system_z:
    :return: numpy lit of array
    """

    origin_xyz = np.array(wgs84_to_ecef(origin_of_local_coordinate_system_x, origin_of_local_coordinate_system_y,
                                                                                    origin_of_local_coordinate_system_z))

    numpy_array_result = []
    poses = np.genfromtxt(path_csv_file, delimiter=',')
    for pose in poses :
        lng, lat, alt, azimuth, tilt, roll = list(pose)

        matrix_angles = get_pose_mat_original_angles([lng, lat, alt, azimuth, tilt, roll])
        global_coordinates = np.array(wgs84_to_ecef(lng,lat,alt))
        local_coordinates =  global_coordinates - origin_xyz
        r = np.concatenate((matrix_angles, np.array([local_coordinates]).transpose()), axis=1)
        r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
        numpy_array_result.append(r)

    return numpy_array_result




def median_errors(ai_competition_result_file_path,ai_competition_gt_file_path,origin_of_local_coordinate_system_x,
                  origin_of_local_coordinate_system_y, origin_of_local_coordinate_system_z):


    ai_competition_result = csc_to_numpy(ai_competition_result_file_path, origin_of_local_coordinate_system_x,
                                         origin_of_local_coordinate_system_y, origin_of_local_coordinate_system_z)

    ai_competition_gt = csc_to_numpy(ai_competition_gt_file_path, origin_of_local_coordinate_system_x,
                                     origin_of_local_coordinate_system_y, origin_of_local_coordinate_system_z)



    if len(ai_competition_gt) == len(ai_competition_result) :

        list_error_on_coordinates = []
        list_error_on_angles = []

        for i, item in enumerate(ai_competition_result) :
            pose_est_i = ai_competition_result[i]
            pose_gt_i = ai_competition_gt[i]
            transl_err = np.linalg.norm(pose_gt_i[0:3, 3] - pose_est_i[0:3, 3])
            list_error_on_coordinates.append(transl_err)
            rot_err = pose_est_i[0:3, 0:3].T.dot(pose_gt_i[0:3, 0:3])
            rot_err = cv2.Rodrigues(rot_err)[0]
            rot_err = np.reshape(rot_err, (1, 3))
            rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
            rot_err = rot_err[0]
            list_error_on_angles.append(rot_err)

        median_error_on_coordinates = np.median(list_error_on_coordinates)
        median_error_on_angles = np.median(list_error_on_angles)
        return median_error_on_coordinates, median_error_on_angles


def scoring(median_error_on_coordinates, median_error_on_angles) :

    if median_error_on_coordinates == 0 :
        score_on_coordinates = 100
    elif median_error_on_coordinates >= 100 :
        score_on_coordinates = 0
    else :
        score_on_coordinates = 100 - median_error_on_coordinates

    if median_error_on_angles == 0 :
        score_on_angle = 100
    elif median_error_on_angles >= 180 :
        score_on_angle = 0
    else :
        score_on_angle = (180 - median_error_on_angles) / 180*100

    overall_score = 0.6 * score_on_coordinates + 0.4 * score_on_angle
    return overall_score




if __name__ == '__main__':
    origin_of_local_coordinate_system_x = 6.5668
    origin_of_local_coordinate_system_y = 46.5191
    origin_of_local_coordinate_system_z = 390

    ai_competition_result_file_path = 'C:\\projects\\vnav\\CrossLoc\\weight\\est.csv'
    ai_competition_gt_file_path = 'C:\\projects\\vnav\\CrossLoc\\weight\\gt.csv'

    median_error_on_coordinates, median_error_on_angles = median_errors(ai_competition_result_file_path,
                                                                        ai_competition_gt_file_path,
                                                                        origin_of_local_coordinate_system_x,
                                                                        origin_of_local_coordinate_system_y,
                                                                        origin_of_local_coordinate_system_z)
    scoring(median_error_on_coordinates,median_error_on_angles)









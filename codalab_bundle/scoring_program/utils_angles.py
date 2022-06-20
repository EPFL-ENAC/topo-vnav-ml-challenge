import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils_reprojection import ecef_to_wgs84,wgs84_to_ecef
from config import settings

def difference_between_angles(a : float, b : float)-> float:
    """
    Compute the shortest difference between two angles
    :param a: angle in degree
    :param b: angle in degree
    :return: angle in degree
    """
    diff_angle = abs(((a - b + 180) % 360) - 180)
    return diff_angle


def rotation_ned_in_ecef(lon, lat):
    """
    Transform longitude and latitude coordinates (in degree) (WGS84, EPSG:4979) into a 3x3 rotation matrix of
    heading-pith-roll NED in ECEF coordinate system
    :param lon: longitude in degree
    :param lat: latitude in degree
    :return: numpy array
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
    assert abs(np.linalg.det(NED) - 1.0) < 1e-6, \
        'NED in NCEF rotation mat. does not have unit determinant, it is: {:.2f}'.format(np.linalg.det(NED))
    return NED


def rotation_matrix_to_euler_angles(pose):
    """
    Extract the roll, tilt, azimuth out of the 4x4 homogeneous extrinsic camera matrix.
    :param R: 4x4 array
    :return: roll, tilt, azimuth
    """
    R = pose

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    degree_values = np.degrees(np.array([x, y, z]))
    degree_values_positive = np.where(degree_values < 0, 360 + degree_values, degree_values)

    return degree_values_positive


def pose_2_sixd_array(pose) -> list:
    """
    Extract the angle information from the 6d coordinate array
    return: x,y,z,azimuth, tilt, roll
    """

    origin_of_local_coordinate_system_x = settings.origin_of_local_coordinate_system_x_wgs84
    origin_of_local_coordinate_system_y = settings.origin_of_local_coordinate_system_y_wgs84
    origin_of_local_coordinate_system_z = settings.origin_of_local_coordinate_system_z_wgs84

    # Pose ECEF local coordinates
    x_local = pose[0, 3]
    y_local = pose[1, 3]
    z_local = pose[2, 3]

    # Transform the center of the local coordinate system from WGS84 to ECEF
    ori_x, ori_y, ori_z = wgs84_to_ecef(
        origin_of_local_coordinate_system_y,
        origin_of_local_coordinate_system_x,
        origin_of_local_coordinate_system_z)


    # Pose ECEF global coordinates
    x = ori_x + x_local
    y = ori_y + y_local
    z = ori_z + z_local

    # Pose WGS84 coordinates
    lat, lng, alt = ecef_to_wgs84(x, y, z)

    # Rotation matrix
    rot_ned_in_ecef = rotation_ned_in_ecef(lng, lat)

    # Remove the last line
    pose = pose[:3]

    # Remove the last column
    pose = pose[:, 0:3]

    # Insert column order
    pose = pose[0:3, [2, 0, 1]]

    # Get the multiplication matrix
    mat_rot_pose = np.matmul(np.linalg.inv(rot_ned_in_ecef), pose)

    # Get the angles
    roll, tilt, azimuth = rotation_matrix_to_euler_angles(mat_rot_pose)

    return lat, lng, alt, x, y, z, azimuth, tilt, roll, x_local, y_local, z_local


def sixd_array_2_pose(sixd_array, coordinate_system ='wgs84'):
    """
    Transform the 6d ndarray, [lat, lon, h, yaw, pitch, roll] into the 4x4 homogeneous extrinsic camera matrix
    """
    if coordinate_system.lower() == 'wgs84' :
        lat, lng, alt, yaw, pitch, roll = sixd_array  # no need to do local conversion when in ECEF
    elif coordinate_system.lower() == 'ecef':
        x, y, z, yaw, pitch, roll = sixd_array
        lon, lat, alt = ecef_to_wgs84(x, y, z)

    rot_ned_in_ecef = rotation_ned_in_ecef(lng, lat)
    rot_pose_in_ned = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
    r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    # transform coordinate system from NED to standard camera sys.
    r = r[0:3, [1, 2, 0]]
    return r




import math
import numpy as np

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


def rotationMatrixToEulerAngles(R):
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
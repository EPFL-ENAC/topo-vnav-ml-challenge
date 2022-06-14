"""
This file aims to evalutate the file uploaded into codalab.
Metrics (median errors, uncertainty) have been defined by the topo laboratory
---
regis.longchamp@epfl.ch
"""

import numpy as np
import cv2
import math
import statistics
import logging

from config import settings
from utils_angles import wgs84_to_ecef,sixd_array_2_pose,difference_between_angles
import json


def check_csv_format(path_csv_estimation, path_csv_ground_truth):
    """
    Analyse the CSV provided to verify its structures
    :param path_csv_estimation:
    :param path_csv_ground_truth:
    :return:
    """

    test_result = {}
    test_result['succeed'] = True
    test_result['details'] = []

    # test 1
    try :
        csv_estimation = np.genfromtxt(path_csv_estimation, delimiter=',')
        csv_ground_truth = np.genfromtxt(path_csv_ground_truth, delimiter=',')
    except :
        test_result['succeed'] = False
        test_result['details'].append('CSV file has not been loaded')


    if test_result.get('succeed') :
        # test 2
        if len(csv_estimation) != len(csv_ground_truth) :
            test_result['succeed'] = False
            test_result['details'].append('The CSV file does not contain enough data. {} line is expected'.format(len(csv_ground_truth)))

        if len(csv_estimation[0]) != len(csv_ground_truth[0])*2 :
            test_result['succeed'] = False
            test_result['details'].append(
                'The CSV file does not contain enough column. {} columns is expected, {} provided'.format(len(csv_ground_truth[0])*2,len(csv_estimation[0])))

    return test_result






def sixd_csv_array_to_pose(path_csv_file):
 """
 Read the 6D position from the CSV file and transform them into numpy array
 :param path_csv_file:
 :return: numpy lit of array
 """

 origin_xyz = np.array(wgs84_to_ecef(settings.origin_of_local_coordinate_system_x_wgs84,
                                     settings.origin_of_local_coordinate_system_y_wgs84,
                                     settings.origin_of_local_coordinate_system_z_wgs84))

 poses = np.genfromtxt(path_csv_file, delimiter=',')

 numpy_array_result = []
 for i,pose in enumerate(poses):
    if len(pose) == 12 : # file submitted by the users (estimation values with 12 columns)
        lng, lat, alt, azimuth, tilt, roll, u_x, u_y, u_z, u_azimith, u_tilt, u_roll = list(pose)
    else : # file ground truth (only 6d pose)
        lng, lat, alt, azimuth, tilt, roll = list(pose)
    matrix_angles = sixd_array_2_pose([lng, lat, alt, azimuth, tilt, roll])
    global_coordinates = np.array(wgs84_to_ecef(lng, lat, alt))
    local_coordinates = global_coordinates - origin_xyz
    r = np.concatenate((matrix_angles, np.array([local_coordinates]).transpose()), axis=1)
    r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
    numpy_array_result.append(r)

 return numpy_array_result



def median_errors(path_csv_estimation : str = settings.path_csv_estimation,
                  path_csv_ground_truth : str = settings.path_csv_ground_truth):
    """
    Compute the median error based on criterias defined by the topo labo.
    :return: median  error on coordinates and angles.
    """
    ai_competition_result = sixd_csv_array_to_pose(path_csv_estimation)
    ai_competition_gt = sixd_csv_array_to_pose(path_csv_ground_truth)

    list_error_on_coordinates = []
    list_error_on_angles = []

    if len(ai_competition_gt) == len(ai_competition_result) :
        for i, item in enumerate(ai_competition_result):
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


def uncertainty(path_csv_estimation : str = settings.path_csv_estimation,
                path_csv_ground_truth : str = settings.path_csv_ground_truth ):
    """
    Compute the uncertainty based on criterias defined by the topo labo.
    :return: uncertainty on coordinates and angles.
    """

    ai_competition_result = np.genfromtxt(path_csv_estimation, delimiter=',')
    ai_competition_ground_truth = np.genfromtxt(path_csv_ground_truth, delimiter=',')

    uncertainty_array_en_x = []
    uncertainty_array_en_y = []
    uncertainty_array_en_z = []
    uncertainty_array_en_azimuth = []
    uncertainty_array_en_tilt = []
    uncertainty_array_en_roll = []

    for i,pose in enumerate(ai_competition_result):
        lng, lat, alt, azimuth_est, tilt_est, roll_est, u_x, u_y, u_z, u_azumith, u_tilt, u_roll = list(pose)
        [x_est,y_est,z_est] = wgs84_to_ecef(lng, lat, alt)

        lng, lat, alt, azimuth_gt, tilt_gt, roll_gt = list(ai_competition_ground_truth[i])
        [x_gt, y_gt, z_gt] = wgs84_to_ecef(lng, lat, alt)

        en_x = float(x_gt - x_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_x, 2))
        uncertainty_array_en_x.append(en_x)
        en_y = float(y_gt - y_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_y, 2))
        uncertainty_array_en_y.append(en_y)
        en_z = float(z_gt - z_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_z, 2))
        uncertainty_array_en_z.append(en_z)
        en_azimuth = difference_between_angles(azimuth_gt,azimuth_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_azumith, 2))
        uncertainty_array_en_azimuth.append(en_azimuth)
        en_tilt = difference_between_angles(tilt_gt, tilt_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_tilt, 2))
        uncertainty_array_en_tilt.append(en_tilt)
        en_roll = difference_between_angles(roll_gt, roll_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_roll, 2))
        uncertainty_array_en_roll.append(en_roll)

    median_uncertainty_en_x = abs(float(np.median(uncertainty_array_en_x)))
    median_uncertainty_en_y = abs(float(np.median(uncertainty_array_en_y)))
    median_uncertainty_en_z = abs(float(np.median(uncertainty_array_en_z)))
    median_uncertainty_en_azimuth = abs(float(np.median(uncertainty_array_en_azimuth)))
    median_uncertainty_en_tilt = abs(float(np.median(uncertainty_array_en_tilt)))
    median_uncertainty_en_roll = abs(float(np.median(uncertainty_array_en_roll)))


    mean_translation = statistics.mean([median_uncertainty_en_x , median_uncertainty_en_y ,median_uncertainty_en_z])
    mean_rotation = statistics.mean([median_uncertainty_en_azimuth , median_uncertainty_en_tilt ,median_uncertainty_en_roll])

    return mean_translation, mean_rotation


def scoring(path_csv_estimation : str = settings.path_csv_estimation,
            path_csv_ground_truth : str = settings.path_csv_ground_truth ):
    """
    This function return the scoring based on the metrics developped by the topo lab.
    :param path_csv_estimation: path of the csv loaded by the user
    :param path_csv_ground_truth: path of the ground truth csv
    :return: the score
    """

    csv_formating_status = check_csv_format(path_csv_estimation, path_csv_ground_truth)
    result = {}
    result['formatting'] = csv_formating_status


    if csv_formating_status.get('succeed') :

        median_error_on_coordinates, median_error_on_angles = median_errors(path_csv_estimation,path_csv_ground_truth)
        uncertainty_on_coordinates, uncertainty_on_angles = uncertainty(path_csv_estimation,path_csv_ground_truth)

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

        if uncertainty_on_coordinates > settings.uncertainty_on_coordinates_min_range  \
                and uncertainty_on_coordinates < settings.uncertainty_on_coordinates_max_range :
            score_u_on_coordinates = 100
        else :
            score_u_on_coordinates = 0

        if uncertainty_on_angles > settings.uncertainty_on_angles_min_range  \
                and uncertainty_on_angles < settings.uncertainty_on_angles_max_range :
            score_u_on_angles = 100
        else :
            score_u_on_angles = 0


        score_error = score_on_coordinates * 0.7 + score_on_angle * 0.3
        result['score_error'] = score_error
        score_uncertainty = score_u_on_coordinates * 0.7 +  score_u_on_angles * 0.3
        result['score_uncertainty'] = score_uncertainty
        score_overall = round(score_error * 0.7 + score_uncertainty * 0.3,3)
        result['score_overall'] = score_overall
        result['median_error_on_coordinates'] = median_error_on_coordinates
        result['median_error_on_angles'] = median_error_on_angles
        result['uncertainty_on_coordinates'] = uncertainty_on_coordinates
        result['uncertainty_on_angles'] = uncertainty_on_angles
        result

    else :
        result['score_error'] = 0
        logging.error('CSV formatting issue')
        logging.error('{}'.format(json.dumps(csv_formating_status)))
        logging.error('path_csv_estimation : {}'.format(json.dumps(path_csv_estimation)))
        logging.error('path_csv_ground_truth : {}'.format(json.dumps(path_csv_ground_truth)))


    return result



if __name__ == '__main__':

    json.dumps(scoring())






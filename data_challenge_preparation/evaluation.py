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
from utils_angles import sixd_array_2_pose,difference_between_angles
from utils_reprojection import ecef_to_wgs84,wgs84_to_ecef
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

        if len(csv_estimation[0]) != settings.nbr_col_without_uncertainty and len(csv_estimation[0]) != settings.nbr_col_with_uncertainty :
            test_result['succeed'] = False
            test_result['details'].append(
                'The CSV file does not contain enough column. {} columns is expected, {} provided'.format(len(csv_ground_truth[0])*2,len(csv_estimation[0])))

    return test_result



def sixd_csv_array_to_pose(path_csv_file):
 """
 Read the 6D position from the CSV file and transform them into numpy array
 :param path_csv_file:
 :return:  dict numpy's array with key = pictures name
 """

 origin_xyz = np.array(wgs84_to_ecef(settings.origin_of_local_coordinate_system_x_wgs84,
                                     settings.origin_of_local_coordinate_system_y_wgs84,
                                     settings.origin_of_local_coordinate_system_z_wgs84))

 poses = np.genfromtxt(path_csv_file, delimiter=',')

 numpy_array_result = {}
 for i,pose in enumerate(poses):
    if len(pose) == settings.nbr_col_with_uncertainty : # file submitted by the users (estimation values with 13 columns)
        pict_name, lng, lat, alt, azimuth, tilt, roll, u_x, u_y, u_z, u_azimith, u_tilt, u_roll = list(pose)
    else : # file ground truth (only 6d pose)
        pict_name, lng, lat, alt, azimuth, tilt, roll = list(pose)
    matrix_angles = sixd_array_2_pose([lng, lat, alt, azimuth, tilt, roll])
    global_coordinates = np.array(wgs84_to_ecef(lng, lat, alt))
    local_coordinates = global_coordinates - origin_xyz
    r = np.concatenate((matrix_angles, np.array([local_coordinates]).transpose()), axis=1)
    r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
    numpy_array_result[pict_name] = r 

 return numpy_array_result



def get_min_uncertainty(path_csv_estimation : str = settings.path_csv_estimation) :
    """
    Return a list of the picture IDs that have
    """

    ai_competition_result = np.genfromtxt(path_csv_estimation, delimiter=',')
    number_of_row = len(ai_competition_result)
    number_of_col = min([len(i) for i in ai_competition_result])
    u_val_best_x_percent = settings.u_val_best_x_percent
    number_of_best_x_percent = round(number_of_row/100*u_val_best_x_percent)
    list_of_best_rows = []
 
    list_score_translation = list()
    list_score_rotation = list()
    dict_score_overall = dict()

    if number_of_col == settings.nbr_col_with_uncertainty :
        for i,pose in enumerate(ai_competition_result):
            picture_name, lng, lat, alt, azimuth_est, tilt_est, roll_est, u_x, u_y, u_z, u_azumith, u_tilt, u_roll = list(pose)
            mean_uncertainty_translation = statistics.mean([u_x , u_y ,u_z])
            mean_uncertainty_rotation = statistics.mean([u_azumith , u_tilt ,u_roll])
            list_score_translation.append((picture_name,mean_uncertainty_translation))
            list_score_rotation.append((picture_name,mean_uncertainty_rotation))

        list_sorted_score_translation = sorted(list_score_translation, key=lambda x: x[1])
        dict_sorted_score_translation = dict((x, rank) for rank, (x, y) in enumerate(list_sorted_score_translation))

        list_sorted_score_rotation = sorted(list_score_rotation, key=lambda x: x[1])
        dict_sorted_score_rotation = dict((x, rank) for rank, (x, y) in enumerate(list_sorted_score_rotation))

        for picture_name, rank_translation in dict_sorted_score_translation.items() :
            rank_rotation = dict_sorted_score_rotation[picture_name]
            overall_rank = 0.7*rank_translation + rank_rotation*0.3
            dict_score_overall[picture_name] = overall_rank

        dict_sorted_score_overall = dict(sorted(dict_score_overall.items(), key = lambda x: x[1]))
        for i, picture_name in enumerate(dict_sorted_score_overall.keys()):
            if i < number_of_best_x_percent :
                list_of_best_rows.append(picture_name)
    
    if number_of_col == settings.nbr_col_without_uncertainty :
        for i,pose in enumerate(ai_competition_result):
            picture_name, lng, lat, alt, azimuth_est, tilt_est, roll_est = list(pose)
            list_of_best_rows.append(picture_name)
    
    return list_of_best_rows
    


def median_errors(path_csv_estimation : str = settings.path_csv_estimation,
                  path_csv_ground_truth : str = settings.path_csv_ground_truth):
    """
    Compute the median error based on criterias defined by the topo labo.
    :return: median  error on coordinates and angles.
    """
    
    list_of_best_rows = get_min_uncertainty(path_csv_estimation)
    
    ai_competition_result = sixd_csv_array_to_pose(path_csv_estimation)
    ai_competition_gt = sixd_csv_array_to_pose(path_csv_ground_truth)  

    list_error_on_coordinates = []
    list_error_on_angles = []

    if len(ai_competition_gt) == len(ai_competition_result) :
        for i, item in ai_competition_result.items():
            if i in list_of_best_rows :
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

    dict_ai_competition_ground_truth = {}
    for i in list(ai_competition_ground_truth) :
        picture_name, lng, lat, alt, azimuth_gt, tilt_gt, roll_gt = i
        dict_ai_competition_ground_truth[int(picture_name)] = [lng, lat, alt, azimuth_gt, tilt_gt, roll_gt]


    if len(ai_competition_result[0]) == 13 :
        for i,pose in enumerate(ai_competition_result):
            picture_name_est, lng, lat, alt, azimuth_est, tilt_est, roll_est, u_x, u_y, u_z, u_azumith, u_tilt, u_roll = list(pose)
            [x_est,y_est,z_est] = wgs84_to_ecef(lat,lng, alt)
            lng, lat, alt, azimuth_gt, tilt_gt, roll_gt = dict_ai_competition_ground_truth.get(int(picture_name_est))
            [x_gt, y_gt, z_gt] = wgs84_to_ecef(lat, lng, alt)
            en_x = abs(float(x_gt - x_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_x, 2)))
            uncertainty_array_en_x.append(en_x)
            en_y = abs(float(y_gt - y_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_y, 2)))
            uncertainty_array_en_y.append(en_y)
            en_z = abs(float(z_gt - z_est) / math.sqrt(pow(settings.u_val_tranlation, 2) + pow(u_z, 2)))
            uncertainty_array_en_z.append(en_z)
            en_azimuth = abs(difference_between_angles(azimuth_gt,azimuth_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_azumith, 2)))
            uncertainty_array_en_azimuth.append(en_azimuth)
            en_tilt = abs(difference_between_angles(tilt_gt, tilt_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_tilt, 2)))
            uncertainty_array_en_tilt.append(en_tilt)
            en_roll = abs(difference_between_angles(roll_gt, roll_est) / math.sqrt(pow(settings.u_val_rotation, 2) + pow(u_roll, 2)))

            uncertainty_array_en_roll.append(en_roll)


        median_uncertainty_en_x = abs(float(np.median(uncertainty_array_en_x)))
        median_uncertainty_en_y = abs(float(np.median(uncertainty_array_en_y)))
        median_uncertainty_en_z = abs(float(np.median(uncertainty_array_en_z)))
        median_uncertainty_en_azimuth = abs(float(np.median(uncertainty_array_en_azimuth)))
        median_uncertainty_en_tilt = abs(float(np.median(uncertainty_array_en_tilt)))
        median_uncertainty_en_roll = abs(float(np.median(uncertainty_array_en_roll)))


        mean_translation = statistics.mean([median_uncertainty_en_x , median_uncertainty_en_y ,median_uncertainty_en_z])
        mean_rotation = statistics.mean([median_uncertainty_en_azimuth , median_uncertainty_en_tilt ,median_uncertainty_en_roll])

    
    else :
        mean_translation = None
        mean_rotation = None

    
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

        score_on_coordinates = settings.scoring_max_point - min(settings.scoring_max_point,median_error_on_coordinates)
        score_on_angle = (settings.scoring_max_angle - min(settings.scoring_max_angle,median_error_on_angles))/ settings.scoring_max_angle*settings.scoring_max_point

        if uncertainty_on_coordinates and uncertainty_on_coordinates > settings.uncertainty_on_coordinates_min_range  \
                and uncertainty_on_coordinates < settings.uncertainty_on_coordinates_max_range :
            score_u_on_coordinates = settings.scoring_max_point
        else :
            score_u_on_coordinates = 0

        if uncertainty_on_angles and uncertainty_on_angles > settings.uncertainty_on_angles_min_range  \
                and uncertainty_on_angles < settings.uncertainty_on_angles_max_range :
            score_u_on_angles = settings.scoring_max_point
        else :
            score_u_on_angles = 0


        score_error = score_on_coordinates * settings.scoring_ratio_on_coordinate + score_on_angle * (1-settings.scoring_ratio_on_coordinate)
        result['score_error'] = score_error
        score_uncertainty = score_u_on_coordinates * settings.scoring_ratio_on_coordinate +  score_u_on_angles * (1-settings.scoring_ratio_on_coordinate)
        result['score_uncertainty'] = score_uncertainty
        score_overall = round(score_error * settings.scoring_ratio_on_error + score_uncertainty * (1-settings.scoring_ratio_on_error),3)
        result['score_overall'] = score_overall
        result['median_error_on_coordinates'] = median_error_on_coordinates
        result['median_error_on_angles'] = median_error_on_angles
        result['uncertainty_on_coordinates'] = uncertainty_on_coordinates
        result['uncertainty_on_angles'] = uncertainty_on_angles


    else :
        result['score_overall'] = 0
        logging.error('CSV formatting issue')
        logging.error('{}'.format(json.dumps(csv_formating_status)))
        logging.error('path_csv_estimation : {}'.format(json.dumps(path_csv_estimation)))
        logging.error('path_csv_ground_truth : {}'.format(json.dumps(path_csv_ground_truth)))


    return result



if __name__ == '__main__':

    print(json.dumps(scoring()))






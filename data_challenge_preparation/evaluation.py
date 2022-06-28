"""
This file aims to score the file uploaded into codalab.
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
    Analyse the  provided CSV to verify its structures (number of columns, format)
    :param path_csv_estimation: path of the estimation CSV file  
    :param path_csv_ground_truth: path of the groundtruth CSV file 
    :return: dictionary containing succeed = True or False
    :rtype: dict
    """
    test_result = {}
    test_result['succeed'] = True
    test_result['details'] = []

    # test 1 : csv format
    try :
        csv_estimation = np.genfromtxt(path_csv_estimation, delimiter=',')
        csv_ground_truth = np.genfromtxt(path_csv_ground_truth, delimiter=',')
    except :
        test_result['succeed'] = False
        test_result['details'].append('CSV file has not been loaded')

    # test 2 : number of columns 
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
 :param path_csv_file: path of the CSV file
 :return:  dictionary :  key = pictures name - value = numpy's array with 4x4 homogeneous extrinsic camera matrix
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



def define_images_subset(path_csv_estimation : str = settings.path_csv_estimation) :
    """
    Return a list of the X best picture IDs (pictures names) based on the uncertainty.
    X depends on the paramter u_val_best_x_percent
    :param path_csv_estimation: path of the estimation CSV file  
    :return: list containing the picture IDs
    """

    estimation_poses = np.genfromtxt(path_csv_estimation, delimiter=',')
    number_of_col = min([len(i) for i in estimation_poses])
    
    # Define the number of picture
    number_of_row = len(estimation_poses)
    number_of_best_x_percent = math.ceil(number_of_row/100*settings.u_val_best_x_percent)
    print('number_of_best_x_percent',number_of_best_x_percent)
 
    list_score_translation = list()
    list_score_rotation = list()
    dict_score_overall = dict()

    if number_of_col == settings.nbr_col_with_uncertainty : # if the CSV provided contains uncertainty columns
        for i,pose in enumerate(estimation_poses):
            picture_name = list(pose)[0]
            u_x, u_y, u_z, u_azumith, u_tilt, u_roll = list(pose)[7:13]
            mean_uncertainty_translation = statistics.mean([u_x , u_y , u_z])
            mean_uncertainty_rotation = statistics.mean([u_azumith , u_tilt ,u_roll])
            list_score_translation.append((picture_name,mean_uncertainty_translation))
            list_score_rotation.append((picture_name,mean_uncertainty_rotation))

        # get the best pictures for the translation
        list_sorted_score_translation = sorted(list_score_translation, key=lambda x: x[1])
        dict_sorted_score_translation = dict((x, rank) for rank, (x, y) in enumerate(list_sorted_score_translation))

        # get the best pictures for the rotation
        list_sorted_score_rotation = sorted(list_score_rotation, key=lambda x: x[1])
        dict_sorted_score_rotation = dict((x, rank) for rank, (x, y) in enumerate(list_sorted_score_rotation))

        # assign an overall score per image for translation + rotation
        for picture_name, rank_translation in dict_sorted_score_translation.items() :
            rank_rotation = dict_sorted_score_rotation[picture_name]
            overall_rank = settings.scoring_ratio_on_coordinate*rank_translation + (1-settings.scoring_ratio_on_coordinate)*rank_rotation
            dict_score_overall[picture_name] = overall_rank

        # select best pictures
        dict_sorted_score_overall = dict(sorted(dict_score_overall.items(), key = lambda x: [1]))
        list_of_best_images = list(dict_sorted_score_overall.keys())[:number_of_best_x_percent]
    
    elif number_of_col == settings.nbr_col_without_uncertainty : # if the CSV provided does not contain uncertainty columns
        list_of_best_images = [ list(pose)[0] for pose in estimation_poses] # the list of all picture is provided 

    return list_of_best_images
    


def median_errors(path_csv_estimation : str = settings.path_csv_estimation,
                  path_csv_ground_truth : str = settings.path_csv_ground_truth):
    """
    Compute the median error based on criterias defined by the topo labo.
    :param path_csv_estimation: path of the estimation CSV file  
    :param path_csv_ground_truth: path of the ground truth CSV file  
    :return: median error on coordinates and angles.
    """
    # Get the list of picture that must be inclued in the median error calcul
    list_of_best_rows = define_images_subset(path_csv_estimation)
    
    # Get the poses
    pose_estimate = sixd_csv_array_to_pose(path_csv_estimation)
    pose_ground_truth = sixd_csv_array_to_pose(path_csv_ground_truth)  

    list_error_on_coordinates = []
    list_error_on_angles = []

    # Compute the median error for position and translation
    if len(pose_ground_truth) == len(pose_estimate) :
        for i, item in pose_estimate.items():
            if i in list_of_best_rows :
                pose_est_i = pose_estimate[i]
                pose_gt_i = pose_ground_truth[i]
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
    Compute the uncertainty based on the Uncertainty quality metric
    :param path_csv_estimation: path of the estimation CSV file  
    :param path_csv_ground_truth: path of the ground truth CSV file  
    :return: uncertainty on coordinates and angles.
    """
    # Get the poses
    ai_competition_result = np.genfromtxt(path_csv_estimation, delimiter=',')
    ai_competition_ground_truth = np.genfromtxt(path_csv_ground_truth, delimiter=',')

    uncertainty_array_en_x = []
    uncertainty_array_en_y = []
    uncertainty_array_en_z = []
    uncertainty_array_en_azimuth = []
    uncertainty_array_en_tilt = []
    uncertainty_array_en_roll = []

    dict_ai_competition_ground_truth = {}
    for pose in list(ai_competition_ground_truth) :
        picture_name, lng, lat, alt, azimuth_gt, tilt_gt, roll_gt = pose
        dict_ai_competition_ground_truth[int(picture_name)] = [lng, lat, alt, azimuth_gt, tilt_gt, roll_gt]

    if len(ai_competition_result[0]) == settings.nbr_col_with_uncertainty : # if the CSV provided contains uncertainty columns
        for i,pose in enumerate(ai_competition_result):
            # estimation 
            picture_name_est, lng, lat, alt, azimuth_est, tilt_est, roll_est, u_x, u_y, u_z, u_azumith, u_tilt, u_roll = list(pose)
            [x_est,y_est,z_est] = wgs84_to_ecef(lat,lng, alt)
            # ground truth
            lng, lat, alt, azimuth_gt, tilt_gt, roll_gt = dict_ai_competition_ground_truth.get(int(picture_name_est))
            [x_gt, y_gt, z_gt] = wgs84_to_ecef(lat, lng, alt)
            # compute en values
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

        # compute median en values
        median_uncertainty_en_x = abs(float(np.median(uncertainty_array_en_x)))
        median_uncertainty_en_y = abs(float(np.median(uncertainty_array_en_y)))
        median_uncertainty_en_z = abs(float(np.median(uncertainty_array_en_z)))
        median_uncertainty_en_azimuth = abs(float(np.median(uncertainty_array_en_azimuth)))
        median_uncertainty_en_tilt = abs(float(np.median(uncertainty_array_en_tilt)))
        median_uncertainty_en_roll = abs(float(np.median(uncertainty_array_en_roll)))

        # compute mean translation and rotation en values
        mean_translation = statistics.mean([median_uncertainty_en_x , median_uncertainty_en_y ,median_uncertainty_en_z])
        mean_rotation = statistics.mean([median_uncertainty_en_azimuth , median_uncertainty_en_tilt ,median_uncertainty_en_roll])

    else :
        mean_translation = None
        mean_rotation = None

    return mean_translation, mean_rotation


def scoring(path_csv_estimation : str = settings.path_csv_estimation,
            path_csv_ground_truth : str = settings.path_csv_ground_truth ):
    """
    This function return the scoring for the subitted csv file.
    :param path_csv_estimation: path of the csv loaded by the user
    :param path_csv_ground_truth: path of the ground truth csv
    :return: dictonary with the overall score + details scores
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
    pass






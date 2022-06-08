import os
import sys

from csv_competition_evaluation import median_errors,scoring


input_dir = sys.argv[1]
output_dir = sys.argv[2]

# input_dir = './in'
# output_dir = './out'


submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')


if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    origin_of_local_coordinate_system_x = 6.5668
    origin_of_local_coordinate_system_y = 46.5191
    origin_of_local_coordinate_system_z = 390


    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    ai_competition_gt_file_path = os.path.join(truth_dir, "gt.csv")
    ai_competition_result_file_path = os.path.join(submit_dir, "est.csv")
    median_error_on_coordinates, median_error_on_angles = median_errors(ai_competition_result_file_path,
                                                                        ai_competition_gt_file_path,
                                                                        origin_of_local_coordinate_system_x,
                                                                        origin_of_local_coordinate_system_y,
                                                                        origin_of_local_coordinate_system_z
                                                                        )

    score = scoring(median_error_on_coordinates, median_error_on_angles)

    output_file.write("correct:%s"%score)
    output_file.close()

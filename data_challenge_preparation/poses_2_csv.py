"""
This file reformat the outputs of the "CrossLoc Benchmark Datasets Setup" tool into CSV.
Input : dataset from https://github.com/TOPO-EPFL/CrossLoc-Benchmark-Datasets
output : CSV file [x, y, z, yaw, pitch, roll]
---
Usage : fill the corresponding section in the config file and run this script. 
---
regis.longchamp@epfl.ch
"""

import os
import numpy as np
import csv

from config import settings
from utils_angles import pose_2_sixd_array



def pose_2_cvs_arrays(coordinate_system : str = 'local' ):
    """
    Convert positions and angles from the pose into WGS84 lat/lng coordinates and ° 
    Gather all poses into arrays, saved in CSV
    param : coordinate_system = ['local','wgs84','ecef']
    """

    path_output_file = settings.path_file_poses_csv

    if settings.path_folder_poses_txt:

        list_of_file = [f for f in os.listdir(settings.path_folder_poses_txt) if
                        os.path.isfile(os.path.join(settings.path_folder_poses_txt, f)) and f.endswith('.txt')]

        row_csv_file = []


        for i, file in enumerate(list_of_file):
            npy_data = np.loadtxt(os.path.join(settings.path_folder_poses_txt,file))
            coordinates = npy_data[0:3, 3]

            lat, lng, alt, x, y, z, azimuth, tilt, roll, x_local, y_local, z_local = pose_2_sixd_array(npy_data)

            if coordinate_system.lower() == 'local' :
                current_row = [x_local, y_local, z_local, azimuth, tilt, roll]
            elif coordinate_system.lower() == 'wgs84' :
                current_row = [lat, lng, alt, azimuth, tilt, roll]
            elif coordinate_system.lower() == 'ecef':
                current_row = [x, y, z, azimuth, tilt, roll]
            else :
                raise Exception("Only 'local' (default), 'wgs84' or 'ecef' coordinate system can be provided")

            row_csv_file.append(current_row)

        with open(path_output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(row_csv_file)


if __name__ == '__main__':
    pose_2_cvs_arrays('wgs84')



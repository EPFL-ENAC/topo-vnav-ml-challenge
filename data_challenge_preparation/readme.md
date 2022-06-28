## Scripts 
* *crossloc_output_2_csv.py* : reformat the outputs of the "CrossLoc localization" tool into CSV.
* *pose_2_csv.py* : reformat the outputs of the "CrossLoc Benchmark Datasets Setup" tool into the 6D pose CSV (x, y, z, yaw, pitch, roll) format.
* *csv_2_pose.py* : reformat the 6D pose CSV (x, y, z, yaw, pitch, roll) into the "CrossLoc Benchmark Datasets Setup" format.
* *utils_angles.py* : matrix operations
* *utils_reprojection.py* : reprojection operations
* *evaluate.py* : used by Codalab to score a submission (based on *evaluation.py*)
* *evaluation.py* : score the submission according to the topo metrics
* *control_csv_pose_processing.py* : control the pipepline pose -> csv -> pose. 

## Usages

To run the scoring process :
* fill the config file in (especially path_csv_ground_truth and path_csv_estimation parameters)
* `run python evaluation.py`  

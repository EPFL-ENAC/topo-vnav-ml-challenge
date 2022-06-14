# Crossloc challenge set up


## Introduction

This repository contains codes to setup a codalab ML competition based on Crossloc inputs and outputs. 
Crossloc is TOPO project that aims to to localize the aerial images by predicting its scene coordinates, and computing the accurate 6D camera pose.


Colalab competion page : [codolab page](https://codalab.lisn.upsaclay.fr/competitions/5227)

Project home page : [here](https://crossloc.github.io/)



## Scripts 
* *crossloc_output_2_csv.py* : reformat the outputs of the "CrossLoc localization" tool into CSV.
* *pose_2_csv.py* : reformat the outputs of the "CrossLoc Benchmark Datasets Setup" tool into the 6D pose CSV (x, y, z, yaw, pitch, roll) format.
* *csv_2_pose.py* : reformat the 6D pose CSV (x, y, z, yaw, pitch, roll) into the "CrossLoc Benchmark Datasets Setup" format.
* *utils_angles.py* : matrix operations
* *utils_reprojection.py* : reprojection operations

## Data
The *data_sample* folder contains a light version of the input data that is used in the current process.

### Crossloc 

**Type** : The crossloc output format is a numpy file. 

**Path** :  /data_sample/from_clossloc/*.npz/<file_name>
where <file_name> is *pose_gt.npy* or *pose_pred.npy*

**Format** : 

| .   | .   | .   | x |
|-----|-----|-----|---|
| .   | .   | .   | y |
| .   | .   | .   | z |
| 0   | 0   | 0   | 1 |

Where the . represent the element of the 3 rotation matrix of heading-pith-roll NED in ECEF coordinate system

x,y,z are in local ECEF coordinate system. 

### CSV output 

**Type** : CSV

**Path** : data_sample/output/*.csv

**Format** : longitude, latitude, altitude, azimuth, tilt, roll


## Usage

To transform Cross poses outputs into one CSV :
* complete the config file
* run `crossloc_output_2_csv.py`




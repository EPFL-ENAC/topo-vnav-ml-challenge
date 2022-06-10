# Crossloc challenge set up


## Project Page 

[codolab page](https://codalab.lisn.upsaclay.fr/competitions/5227)


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




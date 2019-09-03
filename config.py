from pathlib import Path
import os
import datetime
import logging
from sys import platform

# This program does clustering of high resolution raster files using k-means and max-p algorithm

###########################
#### User preferences #####
###########################

param = {}
param["rows"] = 1
param["cols"] = 1
param["region"] = "Europe"
param["minimum_valid"] = 0 # Lowest valid value. Below it, the data is considered NaN
param["kmeans"] = {"min": 50, "max": 150, "step": 10}
param["CRS"] = "epsg:31468"

# Weights for data in files A, B, etc.
param["weights"] = {'A': 1, 'B': 1, 'C': 1, 'D': 1}


# If you want to override the function "identify_number_of_optimum_clusters" that uses the elbow method,
# you can set the number of the maximum clusters for the reference raster yourself. Otherwise, comment the
# following line.
param["max_no_of_cl"] = 200

if platform.startswith('win'):
    # Windows Root Folder
    param["n_jobs"] = 60
elif platform.startswith('linux'):
    # Linux Root Folder
    param["n_jobs"] = -1

###########################
##### Define Paths ########
###########################

fs = os.path.sep

git_RT_folder = os.path.dirname(os.path.abspath(__file__))
root = str(Path(git_RT_folder).parent.parent) + "Database_KS" + fs

paths = {}
region = param["region"]

## Rasters
# At least one of the following lists needs to be non-empty
# The input raster files with mean values (e.g. FLH solar, FLH wind). It must be in .tif format
paths["inputs_mean"] = []
# The input raster files with sum values (e.g. total load). It must be in .tif format
paths["inputs_sum"] = []#[root + "01 Raw inputs" + fs + "Maps" + fs + "Global maps" + fs + "Other" + fs + "HeatDemand_MRM.tif"]
# The input raster files with density values (e.g. load density). It must be in .tif format
paths["inputs_density"] = [root + "01 Raw inputs" + fs + "Maps" + fs + "Global maps" + fs + "Other" + fs + "HeatDemand_MolasseBasin_MWh.tif"]
paths["inputs"] =  paths["inputs_mean"] + paths["inputs_sum"] + paths["inputs_density"]

# Aggregation methods
counter_files = 'A'
param["agg"] = {}
for input_file in paths["inputs"]:
    if input_file in paths["inputs_mean"]:
        param["agg"][counter_files] = mean()
    elif input_file in paths["inputs_sum"]:
        param["agg"][counter_files] = sum()
    elif input_file in paths["inputs_density"]:
        param["agg"][counter_files] = 'density'
    counter_files = chr(ord(counter_files) + 1)

		
## Output Folders
timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
# If you want to use existing folder, input timestamp of that folder in line below and uncomment it.
#timestamp = "20190713T165931"
#timestamp = "20190617T142740"
timestamp = "20190715T125023"
paths["OUT"] = root + "02 Shapefiles for regions" + fs + "Clustering outputs" + fs + region + fs + timestamp + fs
if not os.path.isdir(paths["OUT"]):
    os.mkdir(paths["OUT"])

paths["sub_rasters"] = paths["OUT"] + '01 sub_rasters' + fs
paths["k_means"] = paths["OUT"] + '02 k_means' + fs
paths["polygons"] = paths["OUT"] + '03 polygons' + fs
paths["parts_max_p"] = paths["OUT"] + '04 parts_max_p' + fs
paths["final_output"] = paths["OUT"] + '05 final_output' + fs

try:
    os.makedirs(paths["sub_rasters"])
    os.makedirs(paths["polygons"])
    os.makedirs(paths["parts_max_p"])
    os.makedirs(paths["k_means"])
    os.makedirs(paths["final_output"])
except FileExistsError:
    # directories already exist
    pass
	
	
## Setting basic config for logger.
logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s [%(levelname)s] - %(message)s',
    filename = paths["OUT"] + 'log.txt')  # pass explicit filename here
logger = logging.getLogger()
	
del fs, git_RT_folder, root, region, timestamp
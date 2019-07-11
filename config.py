from pathlib import Path

# This program does clustering of high resolution raster files using k-means and max-p algorithm

###########################
#### User preferences #####
###########################

param = {}
param["rows"] = 5
param["cols"] = 5
param["region"] = "Europe"

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
paths["inputs_sum"] = []
# The input raster files with density values (e.g. load density). It must be in .tif format
paths["inputs_density"] = [root + "01 Raw inputs" + fs + "Maps" + fs + "Global maps" + fs + "Other" + fs + "HeatDemand_MRM.tif"]
paths["inputs"] =  paths["inputs_mean"] + paths["inputs_sum"] + paths["inputs_density"]

# Check whether the inputs are correct
for input_file in paths["inputs"]:
    if not os.path.isfile(input_file):
        return 'file_does_not_exist'
    elif not input_file.endswith('.tif'):
        return 'file_is_not_raster'

		
## Output Folders
timestamp = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
# If you want to use existing folder, input timestamp of that folder in line below and uncomment it.
timestamp = "20190617T142740"
paths["OUT"] = root + "02 Shapefiles for regions" + fs + "Clustering outputs" + fs region + fs + timestamp + fs
if not os.path.isdir(paths["OUT"]):
    os.mkdir(paths["OUT"])

paths["sub_rasters"] = paths["OUT"] + '01 sub_rasters' + fs
paths["polygons"] = paths["OUT"] + '02 polygons' + fs
paths["parts_max_p"] = paths["OUT"] + '03 parts_max_p' + fs
paths["k_means"] = paths["OUT"] + '04 k_means' + fs
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
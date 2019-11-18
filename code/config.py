from pathlib import Path
import os
import datetime
from sys import platform
import numpy as np

def configuration():
    """
    This function is the main configuration function that calls all the other modules in the code.

    :return (paths, param): The dictionary paths containing all the paths to inputs and outputs, and the dictionary param containing all the user preferences.
    :rtype: tuple(dict, dict)
    """
    paths, param = general_settings()
    paths, param = scope_paths_and_parameters(paths, param)

    param = computation_parameters(param)
    param = raster_parameters(param)
    paths, param = raster_cutting_parameters(paths, param)
    param = kmeans_parameters(param)

    paths = output_folders(paths, param)
    paths = output_paths(paths, param)

    return paths, param


def general_settings():
    """
    This function creates and initializes the dictionaries param and paths. It also creates global variables for the root folder ``root``,
    and the system-dependent file separator ``fs``.

    :return (paths, param): The empty dictionary paths, and the dictionary param including some general information.
    :rtype: tuple(dict, dict)
    """
    # These variables will be initialized here, then read in other modules without modifying them.
    global fs
    global root

    param = {}
    param["author"] = "Kais Siala"  # the name of the person running the script
    param["comment"] = "Workshop_example"

    paths = {}
    fs = os.path.sep
    current_folder = os.path.dirname(os.path.abspath(__file__))  
    root = str(Path(current_folder).parent.parent.parent)
    # For use at TUM ENS
    if root[-1] != fs:
        root = root + fs + "Database_KS" + fs
    else:
        root = root + "Database_KS" + fs

    return paths, param

###########################
#### User preferences #####
###########################

def scope_paths_and_parameters(paths, param):
    """
    This function ...

    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return (paths, param): The updated dictionaries paths and param.
    :rtype: tuple(dict, dict)
    """
    # Name tags for the scope
    param["region_name"] = "Ghana"  # Name tag of the spatial scope
    
    # Input rasters with their aggregation function and weights
    inputs = {"Wind FLH": (root + "03 Intermediate files" + fs + "Files Ghana" + fs + "Renewable energy" + fs + "Potential" + fs + "Ghana_WindOn_80_FLH_2015.tif", "mean", 1),
              "Solar weight": (root + "03 Intermediate files" + fs + "Files Ghana" + fs + "Renewable energy" + fs + "Potential" + fs + "Ghana_PV_0_FLH_weight_2015.tif", "sum", 1),
              }
    
    param["raster_names"] = " - ".join(list(inputs.keys()))
    paths["inputs"] = [x[0] for x in list(inputs.values())]
    param["agg"] = [x[1] for x in list(inputs.values())]
    param["weights"] = [x[2] for x in list(inputs.values())]

    return paths, param


def computation_parameters(param):
    """
    This function ...

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    if platform.startswith('win'):
        # Windows Root Folder
        param["n_jobs"] = 60
    elif platform.startswith('linux'):
        # Linux Root Folder
        param["n_jobs"] = -1
        
    return param
    
    
def raster_parameters(param):
    """
    This function ...

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["minimum_valid"] = 0 # Lowest valid value. Below it, the data is considered NaN
    param["CRS"] = "epsg:4326"
    
    return param

 
def raster_cutting_parameters(paths, param):
    """
    This function ...

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["use_shapefile"] = 1
    # If using shapefile, please provide the following parameters/paths
    paths["subregions"] = root + "02 Shapefiles for regions" + fs + "user-defined" + fs + "gadm36_GHA_1.shp"
    param["subregions_name_col"] = "NAME_SHORT"
    # If not using shapefile, please provide the following parameters
    param["rows"] = 2
    param["cols"] = 2
    return paths, param
    

def kmeans_parameters(param):
    """
    This function ...

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["kmeans"] = {"method": "reference_part", # Options: "reference_part" or "maximum_number"
                       "ratio_size_to_std": 7/3,
                       "reference_part": {"min": 50,
                                          "max": 150,
                                          "step": 10},
                       "maximum_number": 400}
    # # If you want to override the function "identify_number_of_optimum_clusters" that uses the elbow method,
    # # you can set the number of the maximum clusters for the reference raster yourself. Otherwise, write 0.
    # param["max_no_of_cl"] = 0
    
    return param
 

###########################
##### Define Paths ########
###########################
		
def output_folders(paths, param):
    """
    This function defines the paths to multiple output folders:
    
      * *region* is the main output folder.
      * ...
      
    All the folders are created at the beginning of the calculation, if they do not already exist,
    
    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
    """
    global root
    global fs

    region = param["region_name"]
    rasters = param["raster_names"]

    # Main output folder
    paths["region"] = root + "02 Shapefiles for regions" + fs + "Clustering outputs" + fs + region + fs + rasters + fs

    # Output folder for sub rasters
    paths["sub_rasters"] = paths["region"] + "01 sub_rasters" + fs
    if not os.path.isdir(paths["sub_rasters"]):
        os.makedirs(paths["sub_rasters"])

    # Output folder for k-means
    paths["k_means"] = paths["region"] + "02 k_means" + fs
    if not os.path.isdir(paths["k_means"]):
        os.makedirs(paths["k_means"])

    # Output folder for polygons
    paths["polygons"] = paths["region"] + "03 polygons" + fs
    if not os.path.isdir(paths["polygons"]):
        os.makedirs(paths["polygons"])

    # Output folder for max-p parts
    paths["parts_max_p"] = paths["region"] + "04 parts_max_p" + fs
    if not os.path.isdir(paths["parts_max_p"]):
        os.makedirs(paths["parts_max_p"])

    # Output folder for final output
    paths["final_output"] = paths["region"] + "05 final_output" + fs
    if not os.path.isdir(paths["final_output"]):
        os.makedirs(paths["final_output"])

    return paths


def output_paths(paths, param):
    """
    """


    # Input statistics
    paths["input_stats"] = paths["region"] + "input_stats.csv"
    paths["non_empty_rasters"] = paths["region"] + "non_empty_rasters.csv"
    paths["kmeans_stats"] = paths["region"] + "kmeans_stats.csv"
    
    # Polygonized clusters after k-means
    paths["polygonized_clusters"] = paths["polygons"] + 'combined_result.shp'
    
    # Combined map after max-p 1
    paths["max_p_combined"] = paths['parts_max_p'] + 'max_p_combined.shp'

    

    return paths
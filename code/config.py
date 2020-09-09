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
    param = maxp_parameters(param)
    param = transmission_parameters(param)

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
    param["author"] = "User"  # the name of the person running the script
    param["comment"] = "Tutorial"

    paths = {}
    fs = os.path.sep
    current_folder = os.path.dirname(os.path.abspath(__file__))
    root = str(Path(current_folder).parent.parent)
    # For use at TUM ENS
    if root[-1] != fs:
        root = root + fs + "Database" + fs
    else:
        root = root + "Database" + fs

    return paths, param


###########################
#### User preferences #####
###########################


def scope_paths_and_parameters(paths, param):
    """
    This function assigns a name for the geographic scope, and collects information regarding the input rasters that will be clustered:
    
    * *region_name* is the name of the geographic scope, which affects the paths where the results are saved.
    * *spatial_scope* is the path to the geographic scope that will be used to clip the map of transmission lines.
      You can ignore it if you are only clustering rasters.
    * *raster_names* are the name tags of the inputs. Preferably, they should not exceed ten (10) characters, as they will be used as attribute names in the created shapefiles.
      If the user chooses strings longer than that, they will be cut to ten characters, and no error is thrown. The name tags are entered as keys into the dictionary ``inputs``.
    * *inputs* are the paths to the input rasters (strings). They are given as the first element of a values tuple for each key in the dictionary ``inputs``.
    * *agg* are the aggregation methods for the input data (strings: either ``'mean'`` or ``'sum'`` or ``'density'``). They are given as the second element of a values tuple for each key in the dictionary ``inputs``.
    * *weights* are the weights of the input data during the clustering (int or float). They are given as the third element of a values tuple for each key in the dictionary ``inputs``.

    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return (paths, param): The updated dictionaries paths and param.
    :rtype: tuple(dict, dict)
    """
    # Name tags for the scope
    param["region_name"] = "Austria"  # Name tag of the spatial scope

    # Path to the shapefile of the scope (useful for lines clustering)
    PathTemp = root + "02 Shapefiles for regions" + fs + "User-defined" + fs
    paths["spatial_scope"] = PathTemp + "gadm36_AUT_0.shp"

    # Input rasters with their aggregation function and weights
    inputs = {
        "Wind_FLH": (
            root + "03 Intermediate files" + fs + "Files Austria" + fs + "Renewable energy" + fs + "Potential" + fs + "Austria_WindOn_80_FLH_2015.tif",
            "mean",
            1,
        ),
        "Solar_FLH": (
            root + "03 Intermediate files" + fs + "Files Austria" + fs + "Renewable energy" + fs + "Potential" + fs + "Austria_PV_0_FLH_2015.tif",
            "mean",
            1,
        ),
    }

    # Input shapefile for transmission lines clustering
    paths["grid_input"] = root + "03 Intermediate files" + fs + "Files Europe" + fs + "Grid" + fs + "grid_cleaned.shp"

    # Do not edit the following lines
    param["raster_names"] = " - ".join(list(inputs.keys()))
    paths["inputs"] = [x[0] for x in list(inputs.values())]
    param["agg"] = [x[1] for x in list(inputs.values())]
    param["weights"] = [x[2] for x in list(inputs.values())]

    return paths, param


def computation_parameters(param):
    """
    This function sets the limit to the number of processes *n_jobs* that can be used in k-means clustering.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    if platform.startswith("win"):
        # Windows Root Folder
        param["n_jobs"] = 60
    elif platform.startswith("linux"):
        # Linux Root Folder
        param["n_jobs"] = -1

    return param


def raster_parameters(param):
    """
    This function sets the parameters for the input rasters.
    
    * *minimum_valid* is the lowest valid value. Below it, the data is considered NaN.
    * *CRS* is the coordinates reference system. It must be the same for all raster input maps.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["minimum_valid"] = 0
    param["CRS"] = "epsg:4326"

    return param


def raster_cutting_parameters(paths, param):
    """
    This function sets how the large input rasters are cut before starting the clustering. There are two options: the maps are either cut using a shapefile of (multi)polygons,
    or using rectangular boxes.
    
    * *use_shapefile*: if 1, a shapefile is used, otherwise rectangular boxes.
    * *subregions*: the path to the shapefile of (multi)polygons that will cut the large raster in smaller parts (only needed if *use_shapefile* is 1).
    * *rows*: number of rows of boxes that the raster will be cut into (only needed if *use_shapefile* is 0).
    * *cols*: number of columns of boxes that the raster will be cut into (only needed if *use_shapefile* is 0).

    :param paths: Dictionary including the paths.
    :type paths: dict
    :param param: Dictionary including the user preferences.
    :type param: dict

    :return (paths, param): The updated dictionaries paths and param.
    :rtype: tuple(dict, dict)
    """
    param["use_shapefile"] = 1

    # If using shapefile, please provide the following parameters/paths
    paths["subregions"] = root + "02 Shapefiles for regions" + fs + "user-defined" + fs + "gadm36_AUT_0.shp"

    # If not using shapefile, please provide the following parameters
    param["rows"] = 2
    param["cols"] = 2

    return paths, param


def kmeans_parameters(param):
    """
    This function sets the parameters for the k-means clustering:
    
    * *method*: Currently, two methods for setting the number of clusters are used. By choosing ``'maximum_number'``, the user sets the total number of clusters for all parts.
      This number will be distributed over the parts depending on their size and the standard deviation of their data. If the user chooses ``'reference_part'``, then the part
      with the highest product of relative size and standard deviation (relatively to the maximum) is chosen as a reference part. For this one, the maximum number of clusters
      is identified using the Elbow method. The optimum number of clusters for all the other parts is a function of that number, and of their relative size and standard deviation.
      
    .. Warning:: The ``'maximum_number'`` might be exceeded due to the rounding of the share of each part.
    
    * *ratio_size_to_std*: This parameter decides about the weight of the relative size and the relative standard deviation (relatively to the maximum) in determining the optimal
      number of clusters for each part. A ratio of 7:3 means that 70% of the weight is on the relative size of the part, and 30% is on its standard deviation. Any number greater
      than zero is accepted.
      
    * *reference_part*: This is a dictionary that contains the parameters for the Elbow method. Cluster sizes between *min* and *max* with a certain *step* will be tested in about
      for-loop, before the optimal number of clusters for the reference part can be identified. The dictionary is only needed it the method is ``'reference_part'``.
      
    * *maximum_number*: This integer sets the maximum number of kmeans clusters for the whole map. It is only used if the method is ``'maximum_number'``.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["kmeans"] = {
        "method": "maximum_number",  # Options: "reference_part" or "maximum_number"
        "ratio_size_to_std": 7 / 3,
        "reference_part": {"min": 50, "max": 150, "step": 10},
        "maximum_number": 300,
    }

    return param


def maxp_parameters(param):
    """
    This function sets the parameters for max-p clustering. Currently, one or two iterations of the max-p algorithm can be used, depending on the number of polygons after
    kmeans.
    
    * *maximum_number*: This number (positive float or integer) defines the maximum number of clusters that the max-p algorithm can cluster in one go. For about 1800 polygons,
      the calculation takes about 8 hours. The problem has a complexity of O(n³) in the Bachmann-Landau notation.
    * *final_number*: This integer defines the number of clusters that the user wishes to obtain at the end. There is no way to force the algorithm to deliver exactly that number
      of regions. However, the threshold can be defined as a function of *final_number*, so that the result will be close to it.
    * *use_results_of_maxp_parts*: This parameter should be set to zero, unless the user has already obtained results for the first run of the max-p algorithm, and want to skip
      it and just run the second round. In that case, the user should set the value at 1.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["maxp"] = {"maximum_number": 1800 * 1.01, "final_number": 9, "use_results_of_maxp_parts": 0}

    return param


def transmission_parameters(param):
    """
    This function sets the parameters for transmission line clustering.
    
    * *CRS_grid*: The coordinates reference system of the shapefile of transmission lines, in order to read it correctly.
    * *default_cap_MVA*: Line capacity in MVA for added lines (to connect electric islands).
    * *default_line_type*: Line type for added lines (to connect electric islands).
    * *number_clusters*: Target number of regions after clustering, to be used as a condition to stop the algorithm.
    * *intermediate_number*: List of numbers of clusters at which an intermediate shapefile will be saved. The values affect the path *grid_intermediate*.
    * *debugging_number*: Number of clusters within an intermediate shapefile, that can be used as an input (for debugging). It affects the path *grid_debugging*.

    :param param: Dictionary including the user preferences.
    :type param: dict

    :return param: The updated dictionary param.
    :rtype: dict
    """
    param["CRS_grid"] = "epsg:4326"
    param["default_cap_MVA"] = 100
    param["default_line_type"] = "AC_OHL"
    param["number_clusters"] = 9
    param["intermediate_number"] = [8200, 4000, 1000, 200, 50]
    param["debugging_number"] = 1000

    return param


###########################
##### Define Paths ########
###########################


def output_folders(paths, param):
    """
    This function defines the paths to multiple output folders:
    
      * *region* is the main output folder. It contains the name of the scope, and the names of the layers used for clustering (as a subfloder).
      * *sub_rasters* is a subfolder containing the parts of the input rasters after cutting them.
      * *k_means* is a subfolder containing the results of the kmeans clustering (rasters).
      * *polygons* is a subfolder containing the polygonized kmeans clusters.
      * *parts_max_p* is a subfolder containing the results of the first round of max-p (if there is a second round).
      * *final_output* is a subfolder containing the final shapefile.
      * *lines_clustering* is a subfolder containing the intermediate and final results of the line clustering.
      
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

    # Output folder for transmission clustering
    paths["lines_clustering"] = (
        root + "02 Shapefiles for regions" + fs + "Clustering outputs" + fs + region + fs + "Clustering transmission lines" + fs
    )
    if not os.path.isdir(paths["lines_clustering"]):
        os.makedirs(paths["lines_clustering"])

    return paths


def output_paths(paths, param):
    """
    This function defines the paths of some of the files that will be saved:
    
      * *input_stats* is the path to a CSV file with general information such as the number of parts, the maximal size and the maximal standard deviation in the parts, and the maximum number
        of clusters as set by the user / by the Elbow method.
      * *non_empty_rasters* is the path to a CSV file with information on each subraster (relative size, standard deviation, etc.).
      * *kmeans_stats* is the path to a CSV file that is only created if the Elbow method is used (i.e. if using a reference part). It contains statistics for kmeans while applying the Elbow method.
      * *polygonized_clusters* is the path to the shapefile with the polygonized rasters for the whole scope.
      * *max_p_combined* is the path to the shapefile that is generated after a first round of the max-p algorithm (if there is a second).
      * *output* is the path to the shapefile that is generated at the end, i.e. after running max_p_whole_map in :mod:`lib.max_p_functions.py`.
    
    For line clustering, the keys start with *grid_*:
    
      * *grid_connected* is the path to the shapefile of lines after adding lines to connect island grids.
      * *grid_clipped* is the path to the shapefile of lines after clipping it with the scope.
      * *grid_voronoi* is the path to the shapefile of voronoi polygons made from the points at the start/end of the lines.
      * *grid_debugging* is the path of an intermediate file during the clustering of regions based on their connectivity.
      * *grid_regions* is the path to the final result of the clustering (shapefile of regions based on their connectivity).
      * *grid_bottlenecks* is the path to the final result of the clustering (shapefile of transmission line bottlenecks).
      
    :param paths: Dictionary including the paths.
    :type paths: dict

    :return paths: The updated dictionary paths.
    :rtype: dict
    """

    # Input statistics
    paths["input_stats"] = paths["region"] + "input_stats.csv"
    paths["non_empty_rasters"] = paths["region"] + "non_empty_rasters.csv"
    paths["kmeans_stats"] = paths["region"] + "kmeans_stats.csv"

    # Polygonized clusters after k-means
    paths["polygonized_clusters"] = paths["polygons"] + "combined_result.shp"

    # Combined map after max-p 1
    paths["max_p_combined"] = paths["parts_max_p"] + "max_p_combined.shp"

    # Final result
    paths["output"] = paths["final_output"] + "final_result.shp"

    # Transmission clustering
    paths["grid_connected"] = paths["lines_clustering"] + "grid_connected.shp"
    paths["grid_clipped"] = paths["lines_clustering"] + "grid_clipped.shp"
    paths["grid_voronoi"] = paths["lines_clustering"] + "grid_voronoi.shp"
    paths["grid_debugging"] = paths["lines_clustering"] + "grid_clusters_" + str(param["debugging_number"]) + ".shp"
    paths["grid_regions"] = paths["lines_clustering"] + "grid_clusters_" + str(param["number_clusters"]) + ".shp"
    paths["grid_bottlenecks"] = paths["lines_clustering"] + "grid_bottlenecks.shp"
    return paths

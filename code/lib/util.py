import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal, osr, ogr
import os
import pysal as ps
from argparse import ArgumentParser
import random as rd
import networkx as nx
import sklearn
from sklearn import cluster
from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString
from shapely.geometry import mapping
from shapely.ops import polygonize
from scipy.spatial import cKDTree
from scipy.optimize import fsolve
import scipy.sparse.csgraph as cg
import shutil
import libpysal
from libpysal.cg.voronoi import voronoi
import math
from math import sqrt, exp
import datetime
import sys
import fiona
import inspect
import rasterio
from rasterio import mask, MemoryFile
import warnings
from warnings import warn
import json


def get_x_y_values(paths):
    """
    This function finds the rel_size and rel_std of the four corners of the x,y scatter plot between rel_size and rel_std.

    :param paths: Dictionary of paths including the path to the CSV file *non_empty_rasters*.
    :type paths: dict
    
    :return: Coordinates of the upper left, upper right, lower left and lower right points of the x,y scatter plot between rel_size and rel_std.
    :rtype: tuple(tuples(int, int))
    """

    # Reading CSV file non_empty_rasters
    df = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])

    # Group by part number, and calculate the product of rel_size and rel_std
    df = df.reset_index(inplace=False)
    df = df.groupby(["part"]).prod()

    # Getting the values of x_low, x_high, y_low, y_high from indices of corners. x -> rel_size and y-> rel_std.
    ul_point = tuple(df.loc[int(df["ul_corner"].idxmax()), ["rel_size", "rel_std"]])
    ur_point = tuple(df.loc[int(df["ur_corner"].idxmax()), ["rel_size", "rel_std"]])
    ll_point = tuple(df.loc[int(df["ll_corner"].idxmax()), ["rel_size", "rel_std"]])
    lr_point = tuple(df.loc[int(df["lr_corner"].idxmax()), ["rel_size", "rel_std"]])

    # In ul_point, x = ul_point[0] and y = ul_point[1]. The same for others.
    return ul_point, ur_point, ll_point, lr_point


def timecheck(*args):
    """
    This function prints information about the progress of the script by displaying the function currently running, and optionally
    an input message, with a corresponding timestamp. If more than one argument is passed to the function, it will raise an exception.

    :param args: Message to be displayed with the function name and the timestamp (optional).
    :type args: string

    :return: The time stamp is printed.
    :rtype: None
    :raise: Too many arguments have been passed to the function, the maximum is only one string.
    """
    if len(args) == 0:
        print(inspect.stack()[1].function + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")) + "\n")

    elif len(args) == 1:
        print(inspect.stack()[1].function + " - " + str(args[0]) + str(datetime.datetime.now().strftime(": %H:%M:%S:%f")) + "\n")

    else:
        raise Exception("Too many arguments have been passed.\nExpected: zero or one \nPassed: " + format(len(args)))


def display_progress(message, progress_stat):
    """
    This function displays a progress bar for long computations. To be used as part of a loop or with multiprocessing.

    :param message: Message to be displayed with the progress bar.
    :type message: string
    :param progress_stat: Tuple containing the total length of the calculation and the current status or progress.
    :type progress_stat: tuple(int, int)

    :return: The status bar is printed.
    :rtype: None
    """
    length = progress_stat[0]
    status = progress_stat[1]
    sys.stdout.write("\r")
    sys.stdout.write(message + " " + "[%-50s] %d%%" % ("=" * ((status * 50) // length), (status * 100) // length))
    sys.stdout.flush()
    if status == length:
        print("\n")


def create_json(filepath, param, param_keys, paths, paths_keys):
    """
    Creates a metadata JSON file containing information about the file in filepath by storing the relevant keys from
    both the param and path dictionaries.

    :param filepath: Path to the file for which the JSON file will be created.
    :type filepath: string
    :param param: Dictionary of dictionaries containing the user input parameters and intermediate outputs.
    :type param: dict
    :param param_keys: Keys of the parameters to be extracted from the *param* dictionary and saved into the JSON file.
    :type param_keys: list of strings
    :param paths: Dictionary of dictionaries containing the paths for all files.
    :type paths: dict
    :param paths_keys: Keys of the paths to be extracted from the *paths* dictionary and saved into the JSON file.
    :type paths_keys: list of strings

    :return: The JSON file will be saved in the desired path *filepath*.
    :rtype: None
    """
    new_file = os.path.splitext(filepath)[0] + ".json"
    new_dict = {}
    # Add standard keys
    param_keys = param_keys + ["author", "comment"]
    for key in param_keys:
        new_dict[key] = param[key]
        if type(param[key]) == np.ndarray:
            new_dict[key] = param[key].tolist()
        if type(param[key]) == tuple:
            param[key] = list(param[key])
            c = 0
            for e in param[key]:
                if type(e) == np.ndarray:
                    new_dict[key][c] = e.tolist()
                c += 1
        if type(param[key]) == dict:
            for k, v in param[key].items():
                if type(v) == np.ndarray:
                    new_dict[key][k] = v.tolist()
                if type(v) == tuple:
                    param[key][k] = list(param[key][k])
                    c = 0
                    for e in param[key][k]:
                        if type(e) == np.ndarray:
                            new_dict[key][k][c] = e.tolist()
                        c += 1
                if type(v) == dict:
                    for k2, v2 in v.items():
                        if type(v2) == np.ndarray:
                            new_dict[key][k][k2] = v2.tolist()
                        if type(v2) == tuple:
                            param[key][k][k2] = list(param[key][k][k2])
                            c = 0
                            for e in param[key][k][k2]:
                                if type(e) == np.ndarray:
                                    new_dict[key][k][k2][c] = e.tolist()
                                c += 1
                        if type(v2) == dict:
                            for k3, v3 in v.items():
                                if type(v3) == np.ndarray:
                                    new_dict[key][k][k2][k3] = v3.tolist()
                                if type(v3) == tuple:
                                    param[key][k][k2][k3] = list(param[key][k][k2][k3])
                                    c = 0
                                    for e in param[key][k][k2][k3]:
                                        if type(e) == np.ndarray:
                                            new_dict[key][k][k2][k3][c] = e.tolist()
                                        c += 1

    for key in paths_keys:
        new_dict[key] = paths[key]
    # Add timestamp
    new_dict["timestamp"] = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    # Add caller function's name
    new_dict["function"] = inspect.stack()[1][3]
    with open(new_file, "w") as json_file:
        json.dump(new_dict, json_file)
    print("files saved: " + new_file)

import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal
from scipy.optimize import fsolve
import os
import pysal as ps
from argparse import ArgumentParser
import random as rd
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
import sklearn
from sklearn import cluster
import shutil
import scipy.sparse.csgraph as cg
import libpysal
from math import sqrt, exp
import datetime
import pprint
import sys
from helping_functions import *


def initialization():
    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + current_date_time)
    
    from config import paths, param, logger
    logger.info('Started at:' + current_date_time)
    
    # Check whether the inputs are correct
    if not len(paths["inputs"]):
        print('no input file given!')
        sys.exit(0)
    for input_file in paths["inputs"]:
        if not os.path.isfile(input_file):
            print('file does not_exist!')
            sys.exit(0)
        elif not input_file.endswith('.tif'):
            print('file is not raster!')
            sys.exit(0)
            
    # Create dataframe for input stats
    df = pd.DataFrame(index=['map_parts_total', 'output_raster_columns', 'output_raster_rows', # from cut_raster_file_to_smaller_boxes
                             'ref_part_name', 'size_max', 'std_max',
                             'max_no_of_cl'],
                      columns=['value'])
    if not os.path.exists(paths["OUT"] + 'input_stats.csv'):
        df.to_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',')
    
    return paths, param, logger
    
    
def cut_raster_file_to_smaller_boxes(param, paths):
    """This function converts the raster file into a m*n boxes with m rows and n columns.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    scale_rows = param["rows"]
    scale_cols = param["cols"]
    logger.info('Cutting raster to smaller boxes.')
    print('------------------------- Cutting raster into smaller parts -------------------------')
    logger.debug('Value of scale_rows is %s', scale_rows)
    logger.debug('Value of scale_columns is %s', scale_cols)

    counter_files = 'A'
    for input_file in paths["inputs"]:
        # Opening the raster file as a dataset
        dataset = gdal.Open(input_file)
        logger.debug('Dealing with raster file = %s', input_file)
        # The number of columns in raster file
        columns_in_raster_file = dataset.RasterXSize
        logger.debug('Columns in raster file = %s', columns_in_raster_file)
        # The number of rows in raster file.
        rows_in_raster_file = dataset.RasterYSize
        logger.debug('Rows in raster file = %s', rows_in_raster_file)

        # no of parts the map will be cut into.
        total_map_parts = scale_rows * scale_cols
        logger.debug('Total parts of map = %s', total_map_parts)
        columns_in_output_raster = int(columns_in_raster_file / scale_cols)
        rows_in_output_raster = int(rows_in_raster_file / scale_rows)

        counter = 1
        gt = dataset.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        
        logger.info('Cutting the raster %s into smaller boxes.', counter_files)
        for i in range(1, scale_cols + 1):
            for j in range(1, scale_rows + 1):
                # cuts the input rasters into n equal parts according to the values assigned as parts_of_map,
                # columns_in_output_raster and rows_in_output_raster. gdal.Translate arguments are:(output_subset_file,
                # input_file, the 4 corners of the square which is to be cut).
                dc = gdal.Translate(paths['sub_rasters'] + counter_files + '_sub_part_%d.tif' % counter, dataset,
                                    projWin=[minx + (i - 1) * columns_in_output_raster * gt[1], maxy - (j - 1) *
                                             rows_in_output_raster * gt[1], minx + columns_in_output_raster * i * gt[1],
                                             maxy - (j * rows_in_output_raster) * gt[1]])
        
                print('Status: Created part: ' + counter_files + '_sub_part_' + str(counter))
                logger.info('Created part: ' + counter_files + '_sub_part_%s', counter)
                counter = counter + 1
        counter_files = chr(ord(counter_files) + 1)

    # Writing the data related to map parts to input_stats.csv file for further use.
    df = pd.read_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',', index_col=0)
    df.loc['map_parts_total', 'value'] = total_map_parts
    df.loc['output_raster_columns', 'value'] = columns_in_output_raster
    df.loc['output_raster_rows', 'value'] = rows_in_output_raster
    logger.info('Writing data in csv file input_stats.csv')
    df.to_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',')
    del dataset
    print('------------------------- == -------------------------')
    
    
def choose_reference_values(param, paths):
    """This function chooses the reference part for the function identify_number_of_optimum_clusters.
    The reference part is chosen based on the product of relative size and relative standard deviation.
    The part with the largest product in all the input files is chosen.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    print('------------------------- Choosing reference values ------------------------- ')

    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + current_date_time)
    logger.info('"choose_reference_values". Started at:' + current_date_time)

    non_empty_rasters = pd.DataFrame(columns=['size', 'std', 'rel_size', 'rel_std', 'prod_size_std'],
                                     index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'file', u'part']))

    # Reading csv file to get total parts of map.
    logger.info('Reading csv file input_stats.csv')
    df = pd.read_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',', index_col=0)
    parts_of_map = int(df.loc['map_parts_total', 'value'])
    logger.debug('Total parts of map:' + str(parts_of_map))

    logger.info('Getting size and std of every raster part.')
    counter_files = 'A'
    for input_file in paths["inputs"]:
        for i in range(1, parts_of_map + 1):
            file = paths["sub_rasters"] + counter_files + '_sub_part_%d.tif' % i
            dataset = gdal.Open(file)
            band_raster = dataset.GetRasterBand(1)
            array_raster = band_raster.ReadAsArray()
            array_raster[array_raster < param["minimum_valid"]] = np.nan
            if np.sum(~np.isnan(array_raster)) == 0:
                continue
        
            array_raster = array_raster.flatten()
            array_raster = array_raster[~np.isnan(array_raster)]
        
            size_raster = len(array_raster)
            std_raster = array_raster.std(axis=0)
        
            logger.debug('Size of part ' + counter_files + str(i) + '=' + str(size_raster))
            logger.debug('Std of part ' + counter_files + str(i) + '=' + str(std_raster))
            non_empty_rasters.loc[(counter_files, i), ['size', 'std']] = (size_raster, std_raster)
        
        logger.info('Calculating relative size, relative std, product of relative size and relative std and four extreme '
                    'corners of data cloud.')
        non_empty_rasters['rel_size'] = non_empty_rasters['size'] / non_empty_rasters['size'].max()
        non_empty_rasters['rel_std'] = non_empty_rasters['std'] / non_empty_rasters['std'].max()
        non_empty_rasters['prod_size_std'] = non_empty_rasters['rel_size'] * non_empty_rasters['rel_std']
        non_empty_rasters['ul_corner'] = np.exp((-non_empty_rasters['rel_size'] + non_empty_rasters['rel_std']).astype(float))
        non_empty_rasters['ur_corner'] = np.exp((non_empty_rasters['rel_size'] + non_empty_rasters['rel_std']).astype(float))
        non_empty_rasters['ll_corner'] = np.exp((-non_empty_rasters['rel_size'] - non_empty_rasters['rel_std']).astype(float))
        non_empty_rasters['lr_corner'] = np.exp((non_empty_rasters['rel_size'] - non_empty_rasters['rel_std']).astype(float))
        counter_files = chr(ord(counter_files) + 1)

    # Writes the numbers of non-empty raster files to csv.
    logger.info('Writing csv file "non_empty_rasters.csv".')
    print('Status: Writing csv file "non_empty_rasters.csv".')
    non_empty_rasters = non_empty_rasters.astype('float64')
    non_empty_rasters.to_csv(paths["OUT"] + 'non_empty_rasters.csv', sep=';', decimal=',')

    # Finding the part with the maximum relative size x relative std.
    logger.info('Finding the part with the maximum relative size x relative std.')
    group_by_part = non_empty_rasters.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(['part']).prod()
    ref_part = group_by_part.loc[group_by_part['prod_size_std'] == group_by_part['prod_size_std'].max()].index.values[0]

    logger.debug('Chosen ref part: ' + str(ref_part) + '.tif')
    print('The chosen reference part is: sub_part_' + str(ref_part) + '.tif')

    # Writing the data related to the reference part to input_stats.csv file for further use.
    df.loc['ref_part_name', 'value'] = ref_part
    df.loc['size_max', 'value'] = non_empty_rasters['size'].max()
    df.loc['std_max', 'value'] = non_empty_rasters['std'].max()
    logger.info('Writing data in csv file input_stats.csv')
    df.to_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"choose_reference_values" finished at: ' + format_time)
	

def identify_number_of_optimum_clusters(param, paths):
    """This function identifies number of optimum clusters which will be chosen for k-means
    Further explanation:
    Standard deviation and size of this reference part are used to estimate the no of clusters of every other part.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    # Read input_stats.csv
    df = pd.read_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',', index_col=0)
    if "max_no_of_cl" in param:
        logger.info('The user has set the maximum number of clusters in the reference raster: %d.', param["max_no_of_cl"])
        df.loc['max_no_of_cl', 'value'] = param["max_no_of_cl"]
        df.to_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',')
    else:
        logger.info('Executing function "identify_number_of_optimum_clusters".')
        print('------------------------- Identifying number of optimum clusters ------------------------- ')
        current_date_time = datetime.datetime.now()
        format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
        print('This part started at: ' + format_time)
        logger.info('"identify_number_of_optimum_clusters" started at: ' + format_time)
        
        ref_part_no = int(df.loc['ref_part_name', 'value'])
        
        counter_files = 'A'
        data = pd.DataFrame(columns=['X', 'Y', 'Value_'+counter_files])
        for input_file in paths["inputs"]:
            reference_part = paths["sub_rasters"] + counter_files + '_sub_part_' + str(ref_part_no) + '.tif'
            logger.debug('Reference part: ' + str(reference_part))
            logger.info('Opening reference part as a dataset.')
            dataset = gdal.Open(reference_part)
            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
            band_raster = dataset.GetRasterBand(1)
            array_raster = band_raster.ReadAsArray()
            array_raster[array_raster <= param["minimum_valid"]] = np.nan
        
            (y_index, x_index) = np.nonzero(~np.isnan(array_raster))
            X = x_index * x_size + upper_left_x + (x_size / 2)
            Y = y_index * y_size + upper_left_y + (y_size / 2)
            array_raster = array_raster.flatten()
            array_raster = array_raster[~np.isnan(array_raster)]
            if len(data)==0:
                data = pd.DataFrame({'X': X, 'Y': Y, 'Value_'+counter_files: array_raster}).set_index(['X', 'Y'])
                counter_files = chr(ord(counter_files) + 1)
            else:
                data = data.join(pd.DataFrame({'X': X, 'Y': Y, 'Value_'+counter_files: array_raster}).set_index(['X', 'Y']), how='inner')
                counter_files = chr(ord(counter_files) + 1)
                
        coef = data.copy()
        coef.reset_index(inplace=True)
        coef['X'] = (coef['X'] - coef['X'].min()) / (coef['X'].max() - coef['X'].min())
        coef['Y'] = (coef['Y'] - coef['Y'].min()) / (coef['Y'].max() - coef['Y'].min())
        n_cols = len(coef.columns[2:])
        # The values in the other columns are given less weight
        for col in coef.columns[2:]:
            if coef[col].min() == coef[col].max():
                coef[col] = 0.1 / np.sqrt(n_cols)
            else:
                coef[col] = 0.1 / np.sqrt(n_cols) * (coef[col] - coef[col].min()) / (coef[col].max() - coef[col].min())
        
        logger.info('Running k-means in order to get the optimum number of clusters.')
        # Only needed to be run once
        k_means_stats = pd.DataFrame(columns=['Inertia', 'Distance', 'Slope'])
        k_min =  param["kmeans"]["min"]
        k_max =  param["kmeans"]["max"]
        k_step = param["kmeans"]["step"]
        k_set = [k_min, k_max]
        k_set.extend(range(k_min+k_step, k_max-k_step+1, k_step))
        for i in k_set:
            print('Checking for cluster number:' + str(i))
            logger.info('Checking for number of clusters = ' + str(i))
            kmeans = sklearn.cluster.KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=1000, tol=0.0001,
                                            precompute_distances='auto', verbose=0, copy_x=True, n_jobs=param["n_jobs"],
                                            algorithm='auto')
            CL = kmeans.fit(coef)
            k_means_stats.loc[i, 'Inertia'] = kmeans.inertia_  # inertia is the sum of the square of the euclidean distances
            logger.debug('Inertia for part: ' + str(i) + ' = ' + str(kmeans.inertia_))
            print('Inertia: ', kmeans.inertia_)
        
            p = OptimumPoint((i - k_min) // k_step + 1, k_means_stats.loc[i, 'Inertia'])
            if i == k_set[0]:
                p1 = p
                k_means_stats.loc[i, 'Distance'] = 0
            elif i == k_set[1]:
                p2 = p
                k_means_stats.loc[i, 'Distance'] = 0
            else:
                k_means_stats.loc[i, 'Distance'] = p.distance_to_line(p1, p2)
                k_means_stats.loc[i, 'Slope'] = k_means_stats.loc[i, 'Distance'] - k_means_stats.loc[i - k_step, 'Distance']
                if abs(k_means_stats.loc[i, 'Slope']) <= 0.2:
                    break
        
        k_means_stats = k_means_stats.astype('float64')
        k_means_stats.to_csv(paths["OUT"] + 'kmeans_stats.csv', index=False, sep=';', decimal=',')
        
        # The point for which the slope is less than threshold is taken as optimum number of clusters.
        maximum_number_of_clusters_ref_part = int(i)
        print('Number of maximum clusters: ' + str(maximum_number_of_clusters_ref_part))
        logger.info('Maximum clusters for reference part: ' + str(maximum_number_of_clusters_ref_part))
        
        # Writing the number of maximum clusters to csv file for further use.
        df.loc['max_no_of_cl', 'value'] = maximum_number_of_clusters_ref_part
        logger.info('Writing data in csv file input_stats.csv')
        df.to_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',')
        
        current_date_time = datetime.datetime.now()
        format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
        print('------------------------- == -------------------------')
        print('This part finished at: ' + format_time)
        print('------------------------- == -------------------------')
        logger.info('"identify_number_of_optimum_clusters" finished at: ' + format_time)
    
    
def k_means_clustering(param, paths):
    """This function does the k-means clustering for every part.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    logger.info('Starting k_means_clustering.')
    print('------------------------- Starting k-means ------------------------- ')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"k_means_clustering" started at: ' + format_time)

    # Reading all necessary inputs from input_stats.csv.
    logger.info('Reading csv file "input_stats.csv".')
    df = pd.read_csv(paths["OUT"] + 'input_stats.csv', sep=';', decimal=',', index_col=0)
    parts_of_map = int(df.loc['map_parts_total', 'value'])
    no_of_columns_in_map = int(df.loc['output_raster_columns', 'value'])
    no_of_rows_in_map = int(df.loc['output_raster_rows', 'value'])
    maximum_no_of_clusters = int(df.loc['max_no_of_cl', 'value'])
    size_max = df.loc['size_max', 'value']
    std_max = df.loc['std_max', 'value']
    
    # Reading the indices of non empty rasters from non_empty_rasters.csv.
    df = pd.read_csv(paths["OUT"] + 'non_empty_rasters.csv', sep=';', decimal=',', index_col=[0,1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))

    # Applying k-means on all parts.
    for i in non_empty_rasters:
        logger.info('Running k-means on part: ' + str(i))
        counter_files = 'A'
        data = pd.DataFrame(columns=['X', 'Y', 'Value_'+counter_files])
        for input_file in paths["inputs"]:
            file = paths["sub_rasters"] + counter_files + '_sub_part_%d.tif' % i
            logger.info('Opening raster file as dataset for conversion to array for k-means.')
            dataset = gdal.Open(file)
            
            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
            band_raster = dataset.GetRasterBand(1)
            array_raster = band_raster.ReadAsArray()
            array_raster[array_raster < param["minimum_valid"]] = np.nan
            
            (y_index, x_index) = np.nonzero(~np.isnan(array_raster))
            X = x_index * x_size + upper_left_x + (x_size / 2)
            Y = y_index * y_size + upper_left_y + (y_size / 2)
            array_raster = array_raster.flatten()
            
            #table = pd.DataFrame({'Value_'+counter_files: array_raster}) ###########################
            
            if len(data)==0:
                table = pd.DataFrame({'Value_'+counter_files: array_raster})
                data = pd.DataFrame({'X': X, 'Y': Y, 'Value_'+counter_files: array_raster[~np.isnan(array_raster)]}).set_index(['X', 'Y'])
                std_of_raster = data['Value_'+counter_files].std(axis=0)
                counter_files = chr(ord(counter_files) + 1)
            else:
                table = pd.DataFrame({'Value_'+counter_files: array_raster})
                data = data.join(pd.DataFrame({'X': X, 'Y': Y, 'Value_'+counter_files: array_raster[~np.isnan(array_raster)]}).set_index(['X', 'Y']), how='inner')
                std_of_raster = max(std_of_raster, data['Value_'+counter_files].std(axis=0))
                counter_files = chr(ord(counter_files) + 1)
            
        coef = data.copy()
        coef.reset_index(inplace=True)
        coef['X'] = (coef['X'] - coef['X'].min()) / (coef['X'].max() - coef['X'].min())
        coef['Y'] = (coef['Y'] - coef['Y'].min()) / (coef['Y'].max() - coef['Y'].min())
        n_cols = len(coef.columns[2:])
        # The values in the other columns are given less weight
        for col in coef.columns[2:]:
            if coef[col].min() == coef[col].max():
                coef[col] = 0.1 / np.sqrt(n_cols)
            else:
                coef[col] = 0.1 / np.sqrt(n_cols) * (coef[col] - coef[col].min()) / (coef[col].max() - coef[col].min())
        size_of_raster = len(coef)

        # this function is used to determine the optimum number of clusters for respective part.
        optimum_no_of_clusters_for_raster = int(np.ceil(maximum_no_of_clusters * (0.7 * (size_of_raster / size_max)
                                                                               + 0.3 * (std_of_raster / std_max))))
        logger.debug('Optimum clusters for part ' + str(i) + ' = ' + str(optimum_no_of_clusters_for_raster))
        logger.debug('70% weight to size and 30% weight to std.')
        if std_of_raster == 0:
            logger.debug('Optimum clusters for this part = 1.')
            optimum_no_of_clusters_for_raster = 1
        if size_of_raster < optimum_no_of_clusters_for_raster:
            logger.debug('Optimum clusters for this part = %d.' % size_of_raster)
            optimum_no_of_clusters_for_raster = size_of_raster

        kmeans = sklearn.cluster.KMeans(n_clusters=optimum_no_of_clusters_for_raster, init='k-means++', n_init=10,
                                        max_iter=1000, tol=0.0001, precompute_distances='auto', verbose=0, copy_x=True,
                                        n_jobs=param["n_jobs"], algorithm='auto')
        CL = kmeans.fit(coef)

        clusters = np.empty([no_of_rows_in_map, no_of_columns_in_map])
        clusters[:] = param["minimum_valid"] - 1
        clusters[y_index, x_index] = CL.labels_
        logger.info('Converting array back to raster. File created: ' + paths["k_means"] +
                    'clusters_part_%d.tif' % i)
        array_to_raster(clusters, paths["k_means"] + 'clusters_part_%d.tif' % i, file)

        table['CL'] = clusters.flatten().astype(int)
        # Calculating the aggregated values for each cluster based on the aggregation method
        counter_files = 'A'
        for input_file in paths["inputs"]:
            if param["agg"][counter_files] == 'density':
                # For pixels with the same size, density = mean
                for cl in table.loc[pd.notnull(table['Value_'+counter_files]), 'CL'].unique():
                    table.loc[table['CL'] == cl, 'Value_'+counter_files] = table.loc[table['CL'] == cl, 'Value_'+counter_files].mean()
            else:
                for cl in table.loc[pd.notnull(table['Value_'+counter_files]), 'CL'].unique():
                    table.loc[table['CL'] == cl, 'Value_'+counter_files] = table.loc[table['CL'] == cl, 'Value_'+counter_files].param["agg"][counter_files]
            # Fill the rest with a value equivalent to NaN in this code
            table.loc[pd.isnull(table['Value_'+counter_files]), 'Value_'+counter_files] = param["minimum_valid"] - 1
            
            counter_files = chr(ord(counter_files) + 1)

        # Group by cluster number, then save the table for later
        table = table.groupby(['CL']).mean()
        table.to_csv(paths["k_means"] + 'clusters_part_%d.csv' % i, index=True, sep=';', decimal=',')
        print('Status: k-means completed for raster part: ' + str(i))

    # End of k-means.
    logger.info('Finished k-means for all parts.')
    print('Status: End of k-means for all parts.')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"k_means_clustering" finished at: ' + format_time)
    
    
def polygonize_after_k_means(param, paths):
    """This function changes from raster after k-means to polygon layers which are used in MaxP algorithm.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    logger.info('Polygonizing all the raster parts obtained after k-means. This is done in order to get all parts ready'
                'for max-p algorithm.')
    print('------------------------- Polygonizing -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"polygonize_after_k_means" started at: ' + format_time)

    # Reading the indices of non empty rasters from non_empty_rasters.csv.
    df = pd.read_csv(paths["OUT"] + 'non_empty_rasters.csv', sep=';', decimal=',', index_col=[0,1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))
    
    for i in non_empty_rasters:
        logger.info('Polygonizing raster part: ' + str(i))
        print('Polygonizing raster part: ' + str(i))
        file_cluster = paths["k_means"] + 'clusters_part_%d.tif' % i
        shape_cluster = paths["polygons"] + 'clusters_part_%d.shp' % i
        polygonize(file_cluster, shape_cluster, 'CL')

        # Read table
        table = pd.read_csv(paths["k_means"] + 'clusters_part_%d.csv' % i, index_col=0, sep=';', decimal=',')
        # Read shapefile with polygons
        file_cluster = gpd.read_file(shape_cluster)
        # Join shapefile and table
        file_cluster.set_index('CL', inplace=True)
        file_cluster = file_cluster.join(table)
        file_cluster.reset_index(inplace=True)
        file_cluster.crs = {'init': param["CRS"]}
        file_cluster.to_file(driver='ESRI Shapefile', filename=paths["polygons"] + 'result_%d.shp' % i)

    # merging all parts together after kmeans to see the output
    logger.info('Merging all parts together.')
    gdf = gpd.read_file(paths["polygons"] + 'result_%d.shp' % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths["polygons"] + 'result_%d.shp' %j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    logger.info('Creating file: combined_result.shp.')
    gdf.crs = {'init': param["CRS"]}
    gdf.to_file(driver='ESRI Shapefile', filename=paths["polygons"] + 'combined_result.shp')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"polygonize_after_k_means" finished at: ' + format_time)


# This class is used in the elbow method to identify the maximum distance between the end point and the start point of
# the curve created between no. of clusters and inertia.
class OptimumPoint:
    def __init__(self, init_x, init_y):
        self.x = init_x
        self.y = init_y

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
        den = sqrt(y_diff ** 2 + x_diff ** 2)
        return num / den


def max_p_algorithm(param, paths):
    """This function applies the max-p algorithm to the obtained polygons.
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    logger.info('Starting max-p.')
    print('------------------------- Max-p One -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"max_p_algorithm" started at: ' + format_time)

    # Reading all necessary inputs from csv files for this function.
    logger.info('Reading csv file: ' + paths["OUT"] + 'non_empty_rasters.csv')
    df = pd.read_csv(paths["OUT"] + 'non_empty_rasters.csv', sep=';', decimal=',', index_col=[0,1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))
    
    # Group by part number, and calculate the product of rel_size and rel_std
    group_by_part = df.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(['part']).prod()

    logger.info('Starting max-p.')
    for i in non_empty_rasters:
        logger.info('Running max-p for part: ' + paths["polygons"] + 'result_%d.shp' % i)
        print('Number of part starting: ', str(i))
        data = gpd.read_file(paths["polygons"] + 'result_%d.shp' % i)
        
        # Calculate the weighted sum in 'Value', that will be used for clustering
        counter_files = 'A'
        data['Value'] = 0
        for input_file in paths["inputs"]:
            scaling = data['Value_' + counter_files].mean()
            data['Value'] = data['Value'] + param["weights"][counter_files] * data['Value_' + counter_files] / scaling
            counter_files = chr(ord(counter_files) + 1)

        logger.info('Creating weights object.')
        w = ps.weights.Queen.from_shapefile(paths["polygons"] + 'result_%d.shp' % i)

        # This loop is used to force any disconnected group of polygons to be assigned to the nearest neighbors
        if len(data) > 1:
            knnw = ps.weights.KNN.from_shapefile(paths["polygons"] + 'result_%d.shp' % i, k=1)
            logger.info('Attaching islands if any to nearest neighbor.')
            w = libpysal.weights.util.attach_islands(w, knnw)

            [n_components, labels] = cg.connected_components(w.sparse)
            if n_components > 1:
                logger.info('Disconnected areas inside the matrix exist. Removing them before max-p can be applied.')
                print('Disconnected areas exist')
                for comp in range(n_components):
                    import pdb; pdb.set_trace()
                    ss = [uu for uu, x in enumerate(labels == comp) if x]
                    dd = data.loc[ss]
                    dd['F'] = 1
                    dd['geometry'] = dd['geometry'].buffer(0)
                    dd = dd.dissolve(by='F')
                    dd.index = [len(data)]
                    dissolve = data.drop(ss)
                    dissolve = dissolve.append(dd)
                    knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                    for cc in range(1, len(data) - 1):
                        countern = 0
                        knn = ps.weights.KNN.from_dataframe(data, k=cc)
                        for s in range(len(ss)):
                            if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                                w.neighbors[ss[s]] = w.neighbors[ss[s]] + knnw.neighbors[len(data)]
                                w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[
                                                                                knnw.neighbors[len(data)][0]] + [
                                                                                ss[s]]
                                countern = countern + 1
                                continue
                        if countern > 0:
                            break
        logger.info('Getting coefficients for threshold equation.')
        coef = get_coefficients(paths)
        logger.debug('Coefficients:', coef)
        print(i, (coef['a'] * (exp(-coef['b'] * (group_by_part.loc[i, 'rel_size'] + (coef['c'] * group_by_part.loc[i, 'rel_std']))))))
        thr = (coef['a'] * (exp(-coef['b'] * (group_by_part.loc[i, 'rel_size'] + (coef['c'] * group_by_part.loc[i, 'rel_std']))))) * data['Value'].sum() * 0.5
        logger.debug('Threshold complete: ' + str(thr))
        # Threshold here was used depending on the size and standard deviation
        if len(data) == 1:
            thr = data['Value'].sum() - 0.01
        #random_no = rd.randint(1000, 1500)  # The range is selected randomly.
        #logger.debug('Random number for seed: ' + str(random_no))
        #np.random.seed(random_no)
        print('Running max-p.')
        logger.info('Running max-p for part: ' + str(i))
        r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
        print('Number of clusters:', end='')
        print(r.p)
        logger.info('Number of clusters after max-p: ' + str(r.p))
        # print('Type:', type(w))
        if r.p == 0:
            import pdb; pdb.set_trace()
            logger.info('No initial solution found.')
            logger.info('Removing disconnected areas again.')
            gal = libpysal.open('%d.gal' % i, 'w')
            gal.write(w)
            gal.close()
            gal = libpysal.open('%d.gal' % i, 'r')
            w = gal.read()
            gal.close()
            [n_components, labels] = cg.connected_components(w.sparse)
            print('Disconnected areas exist again')
            for comp in range(n_components):
                ss = [uu for uu, x in enumerate(labels == comp) if x]
                dd = data.loc[ss]
                dd['F'] = 1
                dd['geometry'] = dd['geometry'].buffer(0)
                dd = dd.dissolve(by='F')
                dd.index = [len(data)]
                dissolve = data.drop(ss)
                dissolve = dissolve.append(dd)
                knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                for cc in range(1, len(data) - 1):
                    countern = 0
                    knn = ps.weights.KNN.from_dataframe(data, k=cc)
                    for s in range(len(ss)):
                        if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                            w.neighbors[str(ss[s])] = w.neighbors[str(ss[s])] + [str(knnw.neighbors[len(data)][0])]
                            w.neighbors[str(knnw.neighbors[len(data)][0])] = w.neighbors[
                                                                                 str(knnw.neighbors[len(data)][0])] + [
                                                                                 str(ss[s])]
                            countern = countern + 1
                            continue
                    if countern > 0:
                        break

            np.random.seed(random_no)
            print('Running max-p again.')
            logger.info('Running max-p again on part: ' + str(i))
            r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
            print('Number of clusters:')
            print(r.p)
            logger.info('Number of clusters after max-p: ' + str(r.p))
        data['CL'] = pd.Series(r.area2region).reindex(data.index)
        data['geometry'] = data['geometry'].buffer(0)
        
        # Calculating the area of each cluster (useful for the density, but needs a projection)
        if param["CRS"] == "epsg:4326":
            data.to_crs(epsg=32662)
            data['area'] = data['geometry'].area / 10**6
            data.to_crs(epsg=4326)
        else:
            data['area'] = data['geometry'].area / 10**6
        
        # Calculating the aggregated values for each cluster based on the aggregation method
        counter_files = 'A'
        for input_file in paths["inputs"]:
            if param["agg"][counter_files] == 'density':
                # First, get the total load for each row
                data['Value_'+counter_files] = data['Value_'+counter_files] * data['area']
                for cl in data.loc[pd.notnull(data['Value_'+counter_files]), 'CL'].unique():
                    data.loc[data['CL'] == cl, 'Value_'+counter_files] = data.loc[data['CL'] == cl, 'Value_'+counter_files].sum() / data.loc[data['CL'] == cl, 'area'].sum()
            else:
                for cl in data.loc[pd.notnull(data['Value_'+counter_files]), 'CL'].unique():
                    data.loc[data['CL'] == cl, 'Value_'+counter_files] = data.loc[data['CL'] == cl, 'Value_'+counter_files].param["agg"][counter_files]
        
        file = data.dissolve(by='CL')
        file.reset_index(inplace=True)

        # Result for every part after max-p one
        logger.info('Creating file: ' + paths['parts_max_p'] + 'max_p_part_%d.shp' % i)
        file.to_file(driver='ESRI Shapefile', filename=paths['parts_max_p'] + 'max_p_part_%d.shp' % i)

    print('------------------------- Merging all parts. -------------------------')

    logger.info('Mering all parts of max-p-1.')
    gdf = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    gdf['geometry'] = gdf.buffer(0)
    logger.info('Creating file: ' + paths['parts_max_p'] + 'max_p_combined.shp')
    gdf.to_file(driver='ESRI Shapefile', filename=paths['parts_max_p'] + 'max_p_combined.shp')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"max_p_algorithm" finished at: ' + format_time)


def max_p_algorithm_2(param, paths):
    """This function runs the max-p algorithm again on the results obtained from max_p_algorithm().
        :param param = The parameters from config.py
        :param paths = The paths to the rasters and to the output folders, from config.py
    """
    logger.info('Starting "max_p_algorithm_2".')
    print('------------------------- Max-p Two -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"max_p_algorithm_2" started at: ' + format_time)

    logger.info('Opening file: ' + paths["parts_max_p"] + 'max_p_combined.shp')
    data = gpd.read_file(paths["parts_max_p"] + 'max_p_combined.shp')
    
    # Calculate the weighted sum in 'Value', that will be used for clustering
    counter_files = 'A'
    data['Value'] = 0
    for input_file in paths["inputs"]:
        scaling = data['Value_' + counter_files].mean()
        data['Value'] = data['Value'] + param["weights"][counter_files] * data['Value_' + counter_files] / scaling
        counter_files = chr(ord(counter_files) + 1)

    logger.info('Creating weights object.')
    w = ps.weights.Queen.from_shapefile(paths["parts_max_p"] + 'max_p_combined.shp')
    knnw = ps.weights.KNN.from_shapefile(paths["parts_max_p"] + 'max_p_combined.shp', k=1)
    w = libpysal.weights.util.attach_islands(w, knnw)
    [n_components, labels] = cg.connected_components(w.sparse)
    print(labels)
    aa = w.islands
    if n_components > 1:
        logger.info('Disconnected areas exist. Removing them for max-p.')
        print('Disconnected areas exist')
        for comp in range(n_components):
            import pdb; pdb.set_trace()
            ss = [uu for uu, x in enumerate(labels == comp) if x]
            dd = data.loc[ss]
            dd['F'] = 1
            dd['geometry'] = dd['geometry'].buffer(0)
            dd = dd.dissolve(by='F')
            dd.index = [len(data)]
            Dissolve = data.drop(ss)
            Dissolve = Dissolve.append(dd)
            knnw = ps.weights.KNN.from_dataframe(Dissolve, k=1)
            for cc in range(1, len(data) - 1):
                countern = 0
                knn = ps.weights.KNN.from_dataframe(data, k=cc)
                for s in range(len(ss)):
                    if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                        w.neighbors[ss[s]] = w.neighbors[ss[s]] + knnw.neighbors[len(data)]
                        w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[knnw.neighbors[len(data)][0]] + [ss[s]]
                        countern = countern + 1
                        continue
                if countern > 0:
                    break
    logger.info('Correcting neighbors.')
    print('Correcting neighbors.')
    w.neighbors = find_neighbors_in_shape_file(paths, w.neighbors)
    print('Neighbors corrected!')
    logger.info('Neighbors corrected.')

    thr = 0.0275 * data['Value'].sum()
    logger.debug('Threshold = ' + str(thr))
    random_no = rd.randint(1000, 1500)  # The range is selected randomly.
    logger.debug('Random number for seed = ' + str(random_no))
    np.random.seed(random_no)

    print('Neighbors Assigned. Running max-p.')
    logger.info('Running max-p.')
    r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
    print('Max-p finished!')
    print('Number of clusters: ' + str(r.p))
    logger.info('Number of clusters: ' + str(r.p))

    data['CL'] = pd.Series(r.area2region).reindex(data.index)
    data['geometry'] = data.buffer(0)
    if args.type == 'mean':
        output = data.dissolve(by='CL', aggfunc='mean')
    elif args.type == 'sum':
        output = data.dissolve(by='CL', aggfunc='sum')
    output.reset_index(inplace=True)
    output['NAME_0'] = 'CL'
    aux = [str(output.loc[i,'CL']).zfill(2) for i in output.index]
    output['NAME_SHORT'] = output['NAME_0'] + aux
    output.crs = {'init': 'epsg:4326'}
    output = output.to_crs(epsg=3034)
    output['Area'] = output.area / 10 ** 6
    logger.info('Creating final output file: ' + folder_names['final_output'] + 'final_result.shp')
    output.to_file(driver='ESRI Shapefile', filename=folder_names['final_output'] + 'final_result.shp')  # Final file

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"max_p_algorithm_2" finished at: ' + format_time)

    return True


def find_neighbors_in_shape_file(paths, existing_neighbors):
    """This function finds the neighbors in the shape file. Somehow, max-p cannot figure out the correct neighbors and
    some clusters are physically neighbors but they are not considered as neighbors. This is where this function comes
    in.
    :param folder_names = The names of all the folders created for output.
    :param existing_neighbors = The neighbors matrix that is created by using w and knn. The new neighbors are to be
                                added to this matrix.
    :
    """
    df = gpd.read_file(paths["parts_max_p"] + 'max_p_combined.shp')
    df["NEIGHBORS"] = None
    for index, cluster_number in df.iterrows():
        # get 'not disjoint' countries
        import pdb; pdb.set_trace()
        neighbors = df[~df.geometry.disjoint(cluster_number.geometry.buffer(0.005))].CL.tolist()
        df1 = df
        df1.crs = {'init': 'epsg:4326'}
        df1 = df1.to_crs({'init': 'epsg:32662'})
        df2 = cluster_number.to_frame().T
        df2 = gpd.GeoDataFrame(df2, geometry='geometry')
        df2.crs = {'init': 'epsg:4326'}
        df2 = df2.to_crs({'init': 'epsg:32662'})
        df2.geometry = df2.geometry.buffer(100)  # in m
        test = gpd.overlay(df1, df2, how='intersection')
        test['area'] = test['geometry'].area / 10 ** 6  # in km²
        test = test[test['area'] > 0.01]  # avoids that neighbors share only a point or a very small area
        neighbors2 = test.CL_1.tolist()
        neighbors = neighbors2
        # remove own name from the list
        neighbors = [cl_no for cl_no in neighbors if cluster_number.CL != cl_no]
        # add names of neighbors as NEIGHBORS value
        df.at[index, "NEIGHBORS"] = ','.join(str(n) for n in neighbors)

    # Making the w.neighbors dictionary for replacing it in max_p_algorithm_2.
    neighbors_corrected = dict()
    for index, row in df.iterrows():
        neighbors_for_one = row['NEIGHBORS'].split(',')
        neighbors_int = list()
        for neighbor in neighbors_for_one:
            if neighbor:
                neighbors_int.append(int(neighbor))
        neighbors_corrected[index] = neighbors_int
        for value in existing_neighbors[index]:
            if value not in neighbors_corrected[index]:
                neighbors_corrected[index].append(value)
        neighbors_corrected[index] = sorted(neighbors_corrected[index])

    return neighbors_corrected





def get_x_y_values(paths):
    """
    This function finds the rel_size and rel_std(coordinates) of the 4 corners of the x,y scatter plot between rel_size
    and rel_std.
    :param folder_names: = The names of all the folders created for output.
    :return: Coordinates of the upper left, upper right, lower left and lower right points of the x,y scatter plot
            between rel_size and rel_std.
    """
    logger.info('Getting coordinates(rel_size,rel_std) of the x,y scatter plot extreme points.')
    logger.info('Reading csv file' + paths["OUT"] + 'non_empty_rasters.csv')
    df = pd.read_csv(paths["OUT"] + 'non_empty_rasters.csv', sep=';', decimal=',', index_col=[0,1])
    # Group by part number, and calculate the product of rel_size and rel_std
    df = df.reset_index(inplace=False)
    df = df.groupby(['part']).prod()

    # Getting the values of x_low, x_high, y_low, y_high from indices of corners. x -> rel_size and y-> rel_std.
    ul_point = tuple(df.loc[int(df['ul_corner'].idxmax()), ['rel_size', 'rel_std']])
    ur_point = tuple(df.loc[int(df['ur_corner'].idxmax()), ['rel_size', 'rel_std']])
    ll_point = tuple(df.loc[int(df['ll_corner'].idxmax()), ['rel_size', 'rel_std']])
    lr_point = tuple(df.loc[int(df['lr_corner'].idxmax()), ['rel_size', 'rel_std']])

    logger.debug('Coordinates of 4 points:')
    logger.debug('ul_point: %s', ul_point)
    logger.debug('ur_point: %s', ur_point)
    logger.debug('ll_point: %s', ll_point)
    logger.debug('lr_point: %s', lr_point)

    # In ul_point, x = ul_point[0] and y = ul_point[1]. The same for others.
    return ul_point, ur_point, ll_point, lr_point


def get_coefficients(paths):
    """
    This function gets the four coefficients A,B and C for solving the 3 equations which will lead to the calculation
    of threshold in max-p-algorithm.
    :param folder_names: = The names of all the folders created for output.
    :return coef: The coefficient values for A,B and C returned as a dictionary.
                  EXPECTED STRUCTURE: {'a': 0.556901762222155, 'b': 2.9138975880272286, 'c': 0.6164969722472001}
    """
    logger.info('Getting coefficients for threshold equation.')
    ul_point, ur_point, ll_point, lr_point = get_x_y_values(paths)
    est = [1, 1, 1]

    logger.info('Running equation solver.')
    coef = fsolve(eq_solver, est, args=(ll_point, ul_point, ur_point), xtol=0.001)

    coef_dict = {'a': coef[0], 'b': coef[1], 'c': coef[2]}
    logger.debug('Coefficients: %s', coef)

    return coef_dict


def eq_solver(coef, ll_point, ul_point, ur_point):
    """
    This function serves as the solver to find coefficient values A,B,C for our defined function which is used to
    calculate the threshold.
    :param coef: The coefficients which are calculated
    :param ll_point: Coordinates of lower left point.
    :param ul_point: Coordinates of upper left point.
    :param ur_point: Coordinates of upper right point.
    :return f: Coefficient values for A, B and C in a numpy array. A is f[0], B is f[1] and C is f[2].
    """
    a = coef[0]
    b = coef[1]
    c = coef[2]

    f = np.zeros(3)

    f[0] = (a * (exp(-b * (ll_point[0] + (c * ll_point[1]))))) - 0.5
    #f[1] = (a * (exp(-b * (ul_point[0] + (c * ul_point[1]))))) - 0.1
    f[2] = (a * (exp(-b * (ur_point[0] + (c * ur_point[1]))))) - 0.01

    return f


if __name__ == '__main__':
    print('------------------------- Starting Clustering -------------------------')
    paths, param, logger = initialization()
    #cut_raster_file_to_smaller_boxes(param, paths)
    #choose_reference_values(param, paths)
    #identify_number_of_optimum_clusters(param, paths)
    #k_means_clustering(param, paths)
    #polygonize_after_k_means(param, paths)
    max_p_algorithm(param, paths)

    #result = max_p_algorithm_2(folders)

    print('----------------------------- END --------------------------------')

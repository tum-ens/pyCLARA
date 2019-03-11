# For details on needed packages and how to install them, check the wiki in clustering repository on GitLab.

import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal, ogr, osr
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
import logging

# Global variables which need to be configured before running the code.
# The command used to run the code: python clustering_code.py --inputfile input_file_path --type wind/solar/load
argument_parser = ArgumentParser(description='This program does clustering of high resolution raster files using '
                                             'k-means and max-p algorithm.')
argument_parser.add_argument('--inputfile', help='The input raster file. It must be in .tif format.', required=True)
argument_parser.add_argument('--rows', help='The number of rows that the input raster will be cut into.')
argument_parser.add_argument('--cols', help='The number of columns that the input raster will be cut into.')
args = argument_parser.parse_args()


def array_to_raster(array, destination_file, input_raster_file):
    """This function changes from array back to raster (used after kmeans algorithm).
    :param array = The array which needs to be converted into a raster.
    :param destination_file = The file name with which the created raster file is saved.
    :param input_raster_file = The original input raster file from which the original coordinates are taken to convert
                               the array back to raster.
    """
    logger.info('Converting array to raster.')

    logger.info('Opening raster file {}'.format(input_raster_file))
    source_raster = gdal.Open(input_raster_file)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = source_raster.GetGeoTransform()

    x_pixels = source_raster.RasterXSize  # number of pixels in x
    logger.debug('x-pixels: %s', x_pixels)
    y_pixels = source_raster.RasterYSize  # number of pixels in y
    logger.debug('y-pixels: %s', y_pixels)
    pixel_size = x_size  # size of the pixel...
    logger.debug('pixel_size : %s', pixel_size)

    x_min = upper_left_x
    y_max = upper_left_y  # x_min & y_max are like the "top left" corner.
    wkt_projection = source_raster.GetProjection()
    driver = gdal.GetDriverByName('GTiff')

    logger.info('Creating file: : %s', destination_file)
    dataset = driver.Create(destination_file, x_pixels, y_pixels, 1, gdal.GDT_Float32, )
    dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.

    return dataset, dataset.GetRasterBand(1)  # If you need to return, remember to return


def polygonize(input_file, output_shape_file, column_name):
    """This function is used to change from a raster to polygons as max-p algorithm only works with polygons.
    :param input_file = The file which needs to be converted to a polygon from a raster.
    :param output_shape_file = The shape file which is generated after polygonization.
    :param column_name = The column name, the values from which are used for conversion.
    """
    logger.info('Polygonizing')

    logger.info('Input file is: : %s', input_file)
    source_raster = gdal.Open(input_file)
    band = source_raster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shape_file):
        logger.info('Deleting data source: : %s', output_shape_file)
        driver.DeleteDataSource(output_shape_file)

    out_data_source = driver.CreateDataSource(output_shape_file + '.shp')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_raster.GetProjectionRef())
    out_layer = out_data_source.CreateLayer(output_shape_file, srs)
    new_field = ogr.FieldDefn(column_name, ogr.OFTReal)
    out_layer.CreateField(new_field)
    logger.info('Creating polygon layer from raster.')
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    out_data_source.Destroy()


# This class is used in the elbow method to identify the maximum distance between the end point and the start point of
# the curve created between no. of clusters and inertia.
class OptimumPoint:
    def __init__(self, init_x, init_y):
        self.x = init_x
        self.y = init_y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
        den = sqrt(y_diff ** 2 + x_diff ** 2)
        return num / den


def create_folders_for_output(input_file, folder_time):
    """This function reads the Input file and creates folders necessary for output.
    :param input_file = File name with a relative path to the script should be given as a string.
                        It should be a high resolution raster(.tif).
    :param folder_time = Time stamp for folder.
    """

    if not os.path.isfile(input_file):
        return 'file_does_not_exist'
    elif not input_file.endswith('.tif'):
        return 'file_is_not_raster'
    else:
        output_sub_rasters = './Results_' + folder_time + '/output_sub_rasters/'
        output_k_means = './Results_' + folder_time + '/output_k_means/'
        output_parts_max_p = './Results_' + folder_time + '/output_parts_max_p/'
        final_output = './Results_' + folder_time + '/final_output/'
        output_polygons = './Results_' + folder_time + '/output_polygons/'
        other_files = './Results_' + folder_time + '/other_files/'
        main_folder = './Results_' + folder_time + '/'

        try:
            os.makedirs(output_sub_rasters)
            os.makedirs(output_k_means)
            os.makedirs(output_parts_max_p)
            os.makedirs(final_output)
            os.makedirs(output_polygons)
            os.makedirs(other_files)
        except FileExistsError:
            # directory already exists
            pass

        folder_names = {'output_sub_rasters': output_sub_rasters, 'output_k_means': output_k_means,
                        'output_parts_max_p': output_parts_max_p, 'final_output': final_output, 'other_files': other_files,
                        'output_polygons': output_polygons, 'main_folder': main_folder}
        return folder_names


def cut_raster_file_to_smaller_boxes(input_file, folder_names, scale_rows=10, scale_columns=10):
    """This function converts the raster file into a m*n boxes with m rows and n columns.
        :param input_file = The input wind file.
        :param scale_rows = Number of rows the raster is to be split in.
        :param scale_columns = Number of columns the raster is to be split in.
        :param folder_names = The names of all the folders created for output.
    """
    logger.info('Cutting raster to smaller boxes.')
    print('------------------------- Cutting raster into smaller parts -------------------------')
    logger.debug('Value of scale_rows is %s', scale_rows)
    logger.debug('Value of scale_columns is %s', scale_columns)
    scale_rows = int(scale_rows)
    scale_columns = int(scale_columns)
    # Opening the raster file as a dataset.
    dataset = gdal.Open(input_file)
    # The number of columns in raster file.
    columns_in_raster_file = dataset.RasterXSize
    logger.debug('Columns in raster file = %s', columns_in_raster_file)
    # The number of rows in raster file.
    rows_in_raster_file = dataset.RasterYSize
    logger.debug('Columns in raster file = %s', columns_in_raster_file)

    row_series = pd.Series(range(1, scale_rows + 1))
    column_series = pd.Series(range(1, scale_columns + 1))

    # no of parts the map will be cut into.
    total_map_parts = scale_rows * scale_columns
    logger.debug('Total parts of map = %s', total_map_parts)

    # no of columns in every output raster after cutting.
    columns_in_output_raster = int(columns_in_raster_file / scale_columns)
    logger.debug('Columns in output raster = %s', columns_in_output_raster)

    # no of rows in  every output raster after cutting
    rows_in_output_raster = int(rows_in_raster_file / scale_rows)
    logger.debug('Rows in output raster = %s', rows_in_output_raster)

    counter = 1
    gt = dataset.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]

    logger.info('Cutting the raster into smaller boxes.')
    for i in column_series.index + 1:
        for j in row_series.index + 1:
            # cuts the input rasters into n equal parts according to the values assigned as parts_of_map,
            # columns_in_output_raster and rows_in_output_raster. gdal.Translate arguments are:(output_subset_file,
            # input_file, the 4 corners of the square which is to be cut).
            dc = gdal.Translate(folder_names['output_sub_rasters'] + 'sub_part_%d.tif' % counter, dataset,
                                projWin=[minx + (i - 1) * columns_in_output_raster * gt[1], maxy - (j - 1) *
                                         rows_in_output_raster * gt[1], minx + columns_in_output_raster * i * gt[1],
                                         maxy - (j * rows_in_output_raster) * gt[1]])

            print('Status: Created part: sub_part_' + str(counter) + '.')
            logger.info('Created part: sub_part_%s', counter)
            counter = counter + 1

    # Writing the data related to map parts to csv file for further use.
    df_vd = pd.DataFrame(data={'map_parts_total': [total_map_parts], 'output_raster_columns': columns_in_output_raster,
                               'output_raster_rows': rows_in_output_raster})
    logger.debug('Created dataframe from parts_of_map.csv %s', df_vd)
    logger.info('Writing csv file parts_of_map.csv')
    df_vd.to_csv(folder_names['other_files'] + 'parts_of_map.csv')
    del dataset

    print('------------------------- == -------------------------')

    return True


def identify_number_of_optimum_clusters(folder_names):
    """This function identifies number of optimum clusters which will be chosen for k-means
    Further explanation:
    Standard deviation and size of this reference part are used to estimate the no of clusters of every other part.
    :param folder_names = The names of all the folders created for output.
    """
    logger.info('Executing function "identify_number_of_optimum_clusters".')
    print('------------------------- Identifying number of optimum clusters ------------------------- ')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"identify_number_of_optimum_clusters" started at: ' + format_time)

    logger.info('Reading csv file "ref_part_and_max_values.csv".')
    ref_part_df = pd.read_csv(folder_names['other_files'] + 'ref_part_and_max_values.csv', index_col=0)
    ref_part_no = ref_part_df['ref_part_name'].values[0]
    reference_part = folder_names['output_sub_rasters'] + 'sub_part_' + str(ref_part_no) + '.tif'
    logger.debug('Reference part: ' + str(reference_part))

    logger.info('Opening reference part as a dataset.')
    dataset = gdal.Open(reference_part)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
    band_raster = dataset.GetRasterBand(1)
    array_raster = band_raster.ReadAsArray()
    array_raster[array_raster <= 0] = np.nan

    (y_index, x_index) = np.nonzero(~np.isnan(array_raster))
    X = x_index * x_size + upper_left_x + (x_size / 2)
    Y = y_index * y_size + upper_left_y + (y_size / 2)
    array_raster = array_raster.flatten()
    array_raster = array_raster[~np.isnan(array_raster)]
    data = pd.DataFrame({'X': X, 'Y': Y, 'Value': array_raster})
    coef = data.copy()
    coef['X'] = (data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min())
    coef['Y'] = (data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min())
    if data['Value'].min() == data['Value'].max():
        coef['Value'] = 0.1
    else:
        coef['Value'] = 0.1 * (data['Value'] - data['Value'].min()) / (data['Value'].max() - data['Value'].min())

    size_ref_part = len(coef)
    std_ref_part = data['Value'].std(axis=0)

    logger.info('Running k-means in order to get the optimum number of clusters.')
    # Only needed to be run once
    k_means_stats = pd.DataFrame(columns=['Inertia', 'Distance', 'Slope'])
    k_set = [50, 150]
    k_set.extend(range(60, 141, 10))
    for i in k_set:
        print('Checking for cluster number:' + str(i))
        logger.info('Checking for number of clusters = ' + str(i))
        kmeans = sklearn.cluster.KMeans(n_clusters=i, init='k-means++', n_init=2, max_iter=1000, tol=0.0001,
                                        precompute_distances='auto', verbose=0, copy_x=True, n_jobs=-1,
                                        algorithm='auto')
        CL = kmeans.fit(coef)
        k_means_stats.loc[i, 'Inertia'] = kmeans.inertia_  # inertia is the sum of the square of the euclidean distances
        logger.debug('Inertia for part: ' + str(i) + ' = ' + str(kmeans.inertia_))
        print('Inertia: ', kmeans.inertia_)

        p = OptimumPoint((i - 50) // 10 + 1, k_means_stats.loc[i, 'Inertia'])
        if i == k_set[0]:
            p1 = p
            k_means_stats.loc[i, 'Distance'] = 0
        elif i == k_set[1]:
            p2 = p
            k_means_stats.loc[i, 'Distance'] = 0
        else:
            k_means_stats.loc[i, 'Distance'] = p.distance_to_line(p1, p2)
            k_means_stats.loc[i, 'Slope'] = k_means_stats.loc[i, 'Distance'] - k_means_stats.loc[i - 10, 'Distance']
            if abs(k_means_stats.loc[i, 'Slope']) <= 0.2:
                break

    k_means_stats.to_csv(folder_names['main_folder'] + 'kmeans_stats.csv')

    # The point for which the slope is less than threshold is taken as optimum number of clusters.
    maximum_number_of_clusters_ref_part = int(i)
    print('Number of maximum clusters: ' + str(maximum_number_of_clusters_ref_part))
    logger.info('Maximum clusters for reference part: ' + str(maximum_number_of_clusters_ref_part))

    # Writing the number of maximum clusters to csv file for further use.
    logger.info('Writing csv file "max_no_of_cl.csv".')
    df_vd = pd.DataFrame(data={'max_no_of_cl': [maximum_number_of_clusters_ref_part]})
    df_vd.to_csv(folder_names['other_files'] + 'max_no_of_cl.csv')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"identify_number_of_optimum_clusters" finished at: ' + format_time)

    return True


# WISHLIST:
# a) check whether we should ceil or floor when estimating the optimal number of clusters
# Possible solution: Check the number after decimal. If it is less than 5, floor, otherwise ceil. But it really doesn't
# matter as it will differ this value by only 1 which is not a big difference.
def k_means_clustering(folder_names):
    """This function does the k-means clustering for every part.
    :param folder_names = The names of all the folders created for output.
    """
    logger.info('Starting k_means_clustering.')
    print('------------------------- Starting k-means ------------------------- ')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"k_means_clustering" started at: ' + format_time)

    # Reading all necessary inputs from different csv files in other_files folder.
    logger.info('Reading csv file "parts_of_map.csv".')
    df = pd.read_csv(folder_names['other_files'] + 'parts_of_map.csv', index_col=0)
    (parts_of_map, no_of_columns_in_map, no_of_rows_in_map) = tuple(df.loc[0])
    logger.info('Reading csv file "non_empty_rasters.csv".')
    df = pd.read_csv(folder_names['main_folder'] + 'non_empty_rasters.csv', index_col=0)
    no_of_parts_of_map = df.index
    logger.info('Reading csv file "max_no_of_cl.csv".')
    df = pd.read_csv(folder_names['other_files'] + 'max_no_of_cl.csv', index_col=0)
    maximum_no_of_clusters = int(df.loc[0, 'max_no_of_cl'])
    logger.info('Reading csv file "ref_part_and_max_values.csv".')
    df = pd.read_csv(folder_names['other_files'] + 'ref_part_and_max_values.csv', index_col=0)
    size_max = df.loc[0, 'size_max']
    std_max = float(df.loc[0, 'std_max'])

    # Applying k-means on all parts.
    for i in no_of_parts_of_map:
        logger.info('Running k-means on part: ' + str(i))
        file = folder_names['output_sub_rasters'] + 'sub_part_%d.tif' % i
        logger.info('Opening raster file as dataset for conversion to array for k-means.')
        dataset = gdal.Open(file)

        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
        band_raster = dataset.GetRasterBand(1)
        array_raster = band_raster.ReadAsArray()
        array_raster[array_raster <= 0] = np.nan

        (y_index, x_index) = np.nonzero(~np.isnan(array_raster))
        X = x_index * x_size + upper_left_x + (x_size / 2)
        Y = y_index * y_size + upper_left_y + (y_size / 2)

        array_raster = array_raster.flatten()
        table = pd.DataFrame({'Value': array_raster})
        data = pd.DataFrame({'X': X, 'Y': Y, 'Value': array_raster[~np.isnan(array_raster)]})
        coef = data.copy()

        coef['X'] = (data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min())
        coef['Y'] = (data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min())
        if data['Value'].min() == data['Value'].max():
            coef['Value'] = 0.1
        else:
            coef['Value'] = 0.1 * (data['Value'] - data['Value'].min()) / (data['Value'].max() - data['Value'].min())

        size_of_raster = len(coef)
        std_of_raster = data['Value'].std(axis=0)

        # this function is used to determine the optimum number of clusters for respective part.
        optimum_no_of_clusters_for_raster = int(np.ceil(maximum_no_of_clusters * (0.7 * (size_of_raster / size_max)
                                                                              + 0.3 * (std_of_raster / std_max))))
        logger.debug('Optimum clusters for part ' + str(i) + ' = ' + str(optimum_no_of_clusters_for_raster))
        logger.debug('70% weight to size and 30% weight to std.')
        if data['Value'].min() == data['Value'].max():
            logger.debug('Optimum clusters for this part = 1.')
            optimum_no_of_clusters_for_raster = 1
        if size_of_raster < optimum_no_of_clusters_for_raster:
            logger.debug('Optimum clusters for this part = %d.' % size_of_raster)
            optimum_no_of_clusters_for_raster = size_of_raster

        kmeans = sklearn.cluster.KMeans(n_clusters=optimum_no_of_clusters_for_raster, init='k-means++', n_init=2,
                                        max_iter=1000, tol=0.0001, precompute_distances='auto', verbose=0, copy_x=True,
                                        n_jobs=-1, algorithm='auto')
        CL = kmeans.fit(coef)

        clusters = np.empty([no_of_rows_in_map, no_of_columns_in_map])
        clusters[:] = -1
        clusters[y_index, x_index] = CL.labels_
        logger.info('Converting array back to raster. File created: ' + folder_names['output_k_means'] +
                    'cluster_part_%d.tif' % i)
        array_to_raster(clusters, folder_names['output_k_means'] + 'cluster_part_%d.tif' % i, file)

        clusters = clusters.flatten()
        table['CL'] = clusters
        for cl in table.loc[pd.notnull(table['Value']), 'CL'].unique():
            table.loc[table['CL'] == cl, 'Value'] = table.loc[table['CL'] == cl, 'Value'].sum()
        table.loc[pd.isnull(table['Value']), 'Value'] = -1
        cluster_means = table['Value'].values.reshape(no_of_rows_in_map, no_of_columns_in_map)
        logger.info('Converting array back to raster. File created: ' + folder_names['output_k_means'] +
                    'value_cluster_part_%d.tif' % i)
        array_to_raster(cluster_means, folder_names['output_k_means'] + 'value_cluster_part_%d.tif' % i, file)
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

    return True


def polygonize_after_k_means(folder_names):
    """This function changes from raster after k-means to polygon layers which are used in MaxP algorithm.
    :param folder_names = The names of all the folders created for output.
    """
    logger.info('Polygonizing all the raster parts obtained after k-means. This is done in order to get all parts ready'
                'for max-p algorithm.')
    print('------------------------- Polygonizing -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"polygonize_after_k_means" started at: ' + format_time)

    logger.info('Reading csv file "non_empty_rasters.csv".')
    non_empty_rasters = pd.read_csv(folder_names['main_folder'] + 'non_empty_rasters.csv', index_col=0)
    for i in non_empty_rasters.index:
        logger.info('Polygonizing raster part: ' + str(i))
        print('Polygonizing raster part: ' + str(i))
        logger.info('Reading files: ')
        logger.info(folder_names['output_k_means'] + 'value_cluster_part_%d.tif' % i)
        logger.info(folder_names['output_k_means'] + 'cluster_part_%d.tif' % i)
        file_value = folder_names['output_k_means'] + 'value_cluster_part_%d.tif' % i
        file_cluster = folder_names['output_k_means'] + 'cluster_part_%d.tif' % i
        shape_value = folder_names['output_polygons'] + 'shapefile_part_%d' % i
        shape_cluster = folder_names['output_polygons'] + 'shapefile_cluster_part_%d' % i
        logger.info('Applying polygonize function on both files.')
        polygonize(file_value, shape_value, 'Value')
        polygonize(file_cluster, shape_cluster, 'CL')

        # There is a need to dissolve both the layers because there is a threshold while dissolving and if the cluster
        # number layer is not dissolved, at the end we will be having less clusters than what we got.
        # Dissolving First Layer
        logger.info('Dissolving first layer.')
        file_value = gpd.read_file(shape_value + '.shp')
        file_value = file_value.dissolve(by='Value')
        file_value.drop(file_value[file_value.index < 0].index, inplace=True)
        file_value.reset_index(inplace=True)

        # Dissolve Second Layer
        logger.info('Dissolving second layer.')
        file_cluster = gpd.read_file(shape_cluster + '.shp')
        file_cluster = file_cluster.dissolve(by='CL')
        file_cluster.drop(file_cluster[file_cluster.index < 0].index, inplace=True)
        file_cluster.reset_index(inplace=True)

        # Join Files Dissolved Files for every layer together
        logger.info('Joining dissolved files.')
        joined_files = gpd.sjoin(file_cluster, file_value, how='left', op='within')
        joined_files.drop(['index_right'], axis=1, inplace=True)
        logger.info('Creating file: ' + folder_names['output_polygons'] + 'result_%d.shp' % i)
        joined_files.to_file(driver='ESRI Shapefile', filename=folder_names['output_polygons'] + 'result_%d.shp' % i)

    # merging all parts together after kmeans to see the output
    logger.info('Merging all parts together.')
    gdf = gpd.read_file(folder_names['output_polygons'] + 'result_%d.shp' % non_empty_rasters.index[0])
    for j in non_empty_rasters.index[1:]:
        gdf_aux = gpd.read_file(folder_names['output_polygons'] + 'result_%d.shp' %j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    logger.info('Creating file: combined_result.shp.')
    gdf.to_file(driver='ESRI Shapefile', filename=folder_names['output_polygons'] + 'combined_result.shp')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"polygonize_after_k_means" finished at: ' + format_time)

    return True


def max_p_algorithm(folder_names):
    """This function applies the max-p algorithm to the obtained polygons.
    :param folder_names = The names of all the folders created for output.
    """
    logger.info('Starting max-p.')
    print('------------------------- Max-p One -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"max_p_algorithm" started at: ' + format_time)

    # Reading all necessary inputs from csv files for this function.
    logger.info('Reading csv file: ' + folder_names['main_folder'] + 'non_empty_rasters.csv')
    non_empty_rasters = pd.read_csv(folder_names['main_folder'] + 'non_empty_rasters.csv', index_col=0)

    logger.info('Starting max-p.')
    for i in non_empty_rasters.index:
        logger.info('Running max-p for part: ' + folder_names['output_polygons'] + 'result_%d.shp' % i)
        print('Number of part starting: ', str(i))
        size = non_empty_rasters.loc[i, 'size']
        std = non_empty_rasters.loc[i, 'std']
        data = gpd.read_file(folder_names['output_polygons'] + 'result_%d.shp' % i)
        data_scaled = MinMaxScaler().fit_transform(data['Value'].values.reshape(-1, 1))

        logger.info('Creating weights object.')
        w = ps.weights.Queen.from_shapefile(folder_names['output_polygons'] + 'result_%d.shp' % i)

        # this loop is used to force any disconnected group of polygons to be assigned to the nearest neighbors
        if len(data) > 1:
            knn1 = ps.weights.KNN.from_shapefile(folder_names['output_polygons'] + 'result_%d.shp' % i, k=1)
            logger.info('Attaching islands if any to nearest neighbor.')
            w = libpysal.weights.util.attach_islands(w, knn1)
            # this is used to force any islands of polygons to be
            # assigned to the nearest neighbors
            aa = w.islands
            [tt, A] = cg.connected_components(w.sparse)
            if tt > 1:
                logger.info('Disconnected areas inside the matrix exist. Removing them before max-p can be applied.')
                print('Disconnected areas exist')
                for gg in range(tt):
                    ss = [uu for uu, x in enumerate(A == gg) if x]
                    dd = data.loc[ss]
                    dd['F'] = 1
                    dd['geometry'] = dd['geometry'].buffer(0)
                    dd = dd.dissolve(by='F')
                    dd.index = [len(data)]
                    dissolve = data.drop(ss)
                    dissolve = dissolve.append(dd)
                    knn1 = ps.weights.KNN.from_dataframe(dissolve, k=1)
                    for cc in range(1, len(data) - 1):
                        countern = 0
                        knn = ps.weights.KNN.from_dataframe(data, k=cc)
                        for s in range(len(ss)):
                            if knn.neighbors[ss[s]][cc - 1] == knn1.neighbors[len(data)][0]:
                                w.neighbors[ss[s]] = w.neighbors[ss[s]] + knn1.neighbors[len(data)]
                                w.neighbors[knn1.neighbors[len(data)][0]] = w.neighbors[
                                                                                knn1.neighbors[len(data)][0]] + [
                                                                                ss[s]]
                                countern = countern + 1
                                continue
                        if countern > 0:
                            break
        logger.info('Getting co-efficients for threshold equation.')
        coef = get_coefficients(folder_names)
        logger.debug('Coefficients:', coef)
        thr = (coef['a'] * (exp(-coef['b'] *
                                (non_empty_rasters.loc[i, 'rel_size'] +
                                 (coef['c'] * non_empty_rasters.loc[i, 'rel_std']))))) * data['Value'].sum() * 0.5
        logger.debug('Threshold complete: ' + str(thr))
        if len(data) == 1:
            thr = data['Value'].sum() - 0.01
        random_no = rd.randint(1000, 1500)  # The range is selected randomly.
        logger.debug('Random number for seed: ' + str(random_no))
        np.random.seed(random_no)
        print('Running max-p.')
        logger.info('Running max-p for part: ' + str(i))
        r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
        print('Number of clusters:', end='')
        print(r.p)
        logger.info('Number of clusters after max-p: ' + str(r.p))
        # print('Type:', type(w))
        if r.p == 0:
            logger.info('No initial solution found.')
            logger.info('Removing in-connected areas again.')
            gal = libpysal.open('%d.gal' % i, 'w')
            gal.write(w)
            gal.close()
            gal = libpysal.open('%d.gal' % i, 'r')
            w = gal.read()
            gal.close()
            [tt, A] = cg.connected_components(w.sparse)
            print('Disconnected areas exist again')
            for gg in range(tt):
                ss = [uu for uu, x in enumerate(A == gg) if x]
                dd = data.loc[ss]
                dd['F'] = 1
                dd['geometry'] = dd['geometry'].buffer(0)
                dd = dd.dissolve(by='F')
                dd.index = [len(data)]
                dissolve = data.drop(ss)
                dissolve = dissolve.append(dd)
                knn1 = ps.weights.KNN.from_dataframe(dissolve, k=1)
                for cc in range(1, len(data) - 1):
                    countern = 0
                    knn = ps.weights.KNN.from_dataframe(data, k=cc)
                    for s in range(len(ss)):
                        if knn.neighbors[ss[s]][cc - 1] == knn1.neighbors[len(data)][0]:
                            w.neighbors[str(ss[s])] = w.neighbors[str(ss[s])] + [str(knn1.neighbors[len(data)][0])]
                            w.neighbors[str(knn1.neighbors[len(data)][0])] = w.neighbors[
                                                                                 str(knn1.neighbors[len(data)][0])] + [
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
        file = data.dissolve(by='CL', aggfunc='sum')
        file.reset_index(inplace=True)

        # Result for every part after max-p one
        logger.info('Creating file: ' + folder_names['output_parts_max_p'] + 'max_p_part_%d.shp' % i)
        file.to_file(driver='ESRI Shapefile', filename=folder_names['output_parts_max_p'] + 'max_p_part_%d.shp' % i)

    print('------------------------- Merging all parts. -------------------------')

    logger.info('Mering all parts of max-p-1.')
    gdf = gpd.read_file(folder_names['output_parts_max_p'] + 'max_p_part_%d.shp' % non_empty_rasters.index[0])
    for j in non_empty_rasters.index[1:]:
        gdf_aux = gpd.read_file(folder_names['output_parts_max_p'] + 'max_p_part_%d.shp' % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    gdf['geometry'] = gdf.buffer(0)
    logger.info('Creating file: ' + folder_names['output_parts_max_p'] + 'max_p_combined.shp')
    gdf.to_file(driver='ESRI Shapefile', filename=folder_names['output_parts_max_p'] + 'max_p_combined.shp')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"max_p_algorithm" finished at: ' + format_time)

    return True


def max_p_algorithm_2(folder_names):
    """This function runs the max-p algorithm again on the results obtained from max_p_algorithm().
    :param folder_names = The names of all the folders created for output.
    """
    logger.info('Starting "max_p_algorithm_2".')
    print('------------------------- Max-p Two -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"max_p_algorithm_2" started at: ' + format_time)

    logger.info('Opening file: ' + folder_names['output_parts_max_p'] + 'max_p_combined.shp')
    data = gpd.read_file(folder_names['output_parts_max_p'] + 'max_p_combined.shp')

    logger.info('Creating weights object.')
    w = ps.weights.Queen.from_shapefile(folder_names['output_parts_max_p'] + 'max_p_combined.shp')
    knn1 = ps.weights.KNN.from_shapefile(folder_names['output_parts_max_p'] + 'max_p_combined.shp', k=1)
    w = libpysal.weights.util.attach_islands(w, knn1)
    [tt, A] = cg.connected_components(w.sparse)
    print(A)
    aa = w.islands
    if tt > 1:
        logger.info('In-connected areas exist. Removing them for max-p.')
        print('In-connected areas exist')
        for gg in range(tt):
            ss = [uu for uu, x in enumerate(A == gg) if x]
            dd = data.loc[ss]
            dd['F'] = 1
            dd['geometry'] = dd['geometry'].buffer(0)
            dd = dd.dissolve(by='F')
            dd.index = [len(data)]
            Dissolve = data.drop(ss)
            Dissolve = Dissolve.append(dd)
            knn1 = ps.weights.KNN.from_dataframe(Dissolve, k=1)
            for cc in range(1, len(data) - 1):
                countern = 0
                knn = ps.weights.KNN.from_dataframe(data, k=cc)
                for s in range(len(ss)):
                    if knn.neighbors[ss[s]][cc - 1] == knn1.neighbors[len(data)][0]:
                        w.neighbors[ss[s]] = w.neighbors[ss[s]] + knn1.neighbors[len(data)]
                        w.neighbors[knn1.neighbors[len(data)][0]] = w.neighbors[knn1.neighbors[len(data)][0]] + [ss[s]]
                        countern = countern + 1
                        continue
                if countern > 0:
                    break
    logger.info('Correcting neighbors.')
    print('Correcting neighbors.')
    w.neighbors = find_neighbors_in_shape_file(folder_names, w.neighbors)
    print('Neighbors corrected!')
    logger.info('Neighbors corrected.')

    thr = 0.0253 * data['Value'].sum()
    logger.debug('Threshold = ' + str(thr))
    random_no = rd.randint(1000, 1500)  # The range is selected randomly.
    logger.debug('Random number for seed = ' + str(random_no))
    np.random.seed(random_no)

    print('Neighbors Assigned. Running max-p.')
    logger.info('Running max-p.')
    r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1,1), floor=thr, floor_variable=data['Value'], initial=5000)
    print('Max-p finished!')
    print('Number of clusters: ' + str(r.p))
    logger.info('Number of clusters: ' + str(r.p))

    data['CL'] = pd.Series(r.area2region).reindex(data.index)
    data['geometry'] = data.buffer(0)
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


def find_neighbors_in_shape_file(folder_names, existing_neighbors):
    """This function finds the neighbors in the shape file. Somehow, max-p cannot figure out the correct neighbors and
    some clusters are physically neighbors but they are not considered as neighbors. This is where this function comes
    in.
    :param folder_names = The names of all the folders created for output.
    :param existing_neighbors = The neighbors matrix that is created by using w and knn. The new neighbors are to be
                                added to this matrix.
    :
    """
    df = gpd.read_file(folder_names['output_parts_max_p'] + 'max_p_combined.shp')
    df["NEIGHBORS"] = None
    for index, cluster_number in df.iterrows():
        # get 'not disjoint' countries
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


def choose_reference_values(folder_names):
    """This function chooses the reference part for the function identify_number_of_optimum_clusters.
    The reference part is chosen based on size. The part with the largest size is chosen.
    The reference standard deviation is calculated by taking the average of standard deviation from all parts.
    :param folder_names = The names of all the folders created for output.
    """
    print('------------------------- Choosing reference values ------------------------- ')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"choose_reference_values". Started at:' + format_time)

    non_empty_rasters = pd.DataFrame(columns=['size', 'std', 'rel_size', 'rel_std', 'prod_size_std', 'ratio_size_std'])

    # Reading csv file to get total parts of map.
    logger.info('Reading csv file parts_of_map.csv')
    map_df = pd.read_csv(folder_names['other_files'] + 'parts_of_map.csv', index_col=0)
    map_parts = map_df['map_parts_total'].values[0]
    parts_of_map = int(map_parts)
    logger.debug('Total parts of map:' + str(parts_of_map))

    no_of_parts_of_map = pd.Series(range(1, parts_of_map))

    logger.info('Getting size and std of every raster part.')
    for i in no_of_parts_of_map.index + 1:
        file = folder_names['output_sub_rasters'] + 'sub_part_%d.tif' % i
        dataset = gdal.Open(file)
        band_raster = dataset.GetRasterBand(1)
        array_raster = band_raster.ReadAsArray()
        array_raster[array_raster <= 0] = np.nan
        if np.sum(~np.isnan(array_raster)) == 0:
            continue

        array_raster = array_raster.flatten()
        array_raster = array_raster[~np.isnan(array_raster)]

        size_raster = len(array_raster)
        std_raster = array_raster.std(axis=0)

        logger.debug('Size of part ' + str(i) + '=' + str(size_raster) + '.')
        logger.debug('Std of part ' + str(i) + '=' + str(std_raster) + '.')
        non_empty_rasters.loc[i, ['size', 'std']] = (size_raster, std_raster)

    logger.info('Calculating relative size, relative std, product of relative size and relative std and four extreme '
                'corners of data cloud.')
    non_empty_rasters['rel_size'] = non_empty_rasters['size'] / non_empty_rasters['size'].max()
    non_empty_rasters['rel_std'] = non_empty_rasters['std'] / non_empty_rasters['std'].max()
    non_empty_rasters['prod_size_std'] = non_empty_rasters['rel_size'] * non_empty_rasters['rel_std']
    non_empty_rasters['ul_corner'] = np.exp((-non_empty_rasters['rel_size'] + non_empty_rasters['rel_std']).astype(float))
    non_empty_rasters['ur_corner'] = np.exp((non_empty_rasters['rel_size'] + non_empty_rasters['rel_std']).astype(float))
    non_empty_rasters['ll_corner'] = np.exp((-non_empty_rasters['rel_size'] - non_empty_rasters['rel_std']).astype(float))
    non_empty_rasters['lr_corner'] = np.exp((non_empty_rasters['rel_size'] - non_empty_rasters['rel_std']).astype(float))

    # Writes the numbers of non-empty raster files to csv.
    logger.info('Writing csv file "non_empty_rasters.csv".')
    print('Status: Writing csv file "non_empty_rasters.csv".')
    non_empty_rasters.to_csv(folder_names['main_folder'] + 'non_empty_rasters.csv')

    # Finding the part with the maximum relative size x relative std.
    logger.info('Finding the part with the maximum relative size x relative std.')
    ref_part = \
        non_empty_rasters.loc[
            non_empty_rasters['prod_size_std'] == non_empty_rasters['prod_size_std'].max()].index.values[
            0]

    logger.debug('Chosen ref part: ' + str(ref_part) + '.tif')
    print('The chosen reference part is: sub_part_' + str(ref_part) + '.tif')

    # Writing the values needed to a csv file in order to make this part independent of others.
    logger.info('Writing csv file "ref_part_and_max_values.csv".')
    df_vd = pd.DataFrame(data={'ref_part_name': [str(ref_part)], 'size_max': [non_empty_rasters['size'].max()],
                               'std_max': [non_empty_rasters['std'].max()]})
    df_vd.to_csv(folder_names['other_files'] + 'ref_part_and_max_values.csv')

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"choose_reference_values" finished at: ' + format_time)

    return True


def get_x_y_values(folder_names):
    """
    This function finds the rel_size and rel_std(coordinates) of the 4 corners of the x,y scatter plot between rel_size
    and rel_std.
    :param folder_names: = The names of all the folders created for output.
    :return: Coordinates of the upper left, upper right, lower left and lower right points of the x,y scatter plot
            between rel_size and rel_std.
    """
    logger.info('Getting coordinates(rel_size,rel_std) of the x,y scatter plot extreme points.')
    logger.info('Reading csv file' + folder_names['main_folder'] + 'non_empty_rasters.csv')
    non_empty_rasters_df = pd.read_csv(folder_names['main_folder'] + 'non_empty_rasters.csv', index_col=0)

    # Getting the indices of the four points identified as corners of the x,y scatter plot between rel_size and rel_std.
    ul_corner_index = int(non_empty_rasters_df['ul_corner'].idxmax())
    ur_corner_index = int(non_empty_rasters_df['ur_corner'].idxmax())
    ll_corner_index = int(non_empty_rasters_df['ll_corner'].idxmax())
    lr_corner_index = int(non_empty_rasters_df['lr_corner'].idxmax())

    # Getting the values of x_low, x_high, y_low, y_high from indices of corners. x -> rel_size and y-> rel_std.
    ul_point = tuple(non_empty_rasters_df.loc[ul_corner_index, ['rel_size', 'rel_std']])
    ur_point = tuple(non_empty_rasters_df.loc[ur_corner_index, ['rel_size', 'rel_std']])
    ll_point = tuple(non_empty_rasters_df.loc[ll_corner_index, ['rel_size', 'rel_std']])
    lr_point = tuple(non_empty_rasters_df.loc[lr_corner_index, ['rel_size', 'rel_std']])

    logger.debug('Coordinates of 4 points:')
    logger.debug('ul_point: %s', ul_point)
    logger.debug('ur_point: %s', ur_point)
    logger.debug('ll_point: %s', ll_point)
    logger.debug('lr_point: %s', lr_point)

    # In ul_point, x = ul_point[0] and y = ul_point[1]. The same for others.

    return ul_point, ur_point, ll_point, lr_point


def get_coefficients(folder_names):
    """
    This function gets the four coefficients A,B and C for solving the 3 equations which will lead to the calculation
    of threshold in max-p-algorithm.
    :param folder_names: = The names of all the folders created for output.
    :return coef: The coefficient values for A,B and C returned as a dictionary.
                  EXPECTED STRUCTURE: {'a': 0.556901762222155, 'b': 2.9138975880272286, 'c': 0.6164969722472001}
    """
    logger.info('Getting coefficients for threshold equation.')
    ul_point, ur_point, ll_point, lr_point = get_x_y_values(folder_names)
    est = [1, 1, 1]

    logger.info('Running equation solver.')
    coef = fsolve(eq_solver, est, args=(ll_point, ul_point, ur_point), xtol=0.001)

    coef_dict = {'a': coef[0], 'b': coef[1], 'c': coef[2]}
    #print(coef_dict)
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
    f[1] = (a * (exp(-b * (ul_point[0] + (c * ul_point[1]))))) - 0.1
    f[2] = (a * (exp(-b * (ur_point[0] + (c * ur_point[1]))))) - 0.01

    return f


if __name__ == '__main__':
    print('------------------------- Starting Clustering -------------------------')
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('Started at: ' + formatted_time)

    time_folder = current_time.strftime("%Y-%m-%d_%H%M%S")
    # If you want to use existing folder, input timestamp of that folder in line below and uncomment it.
    time_folder = '2019-03-07_174110'
    folders = create_folders_for_output(args.inputfile, time_folder)

    #  Setting basic config for logger.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=folders['main_folder'] + 'log.txt')  # pass explicit filename here
    logger = logging.getLogger()

    # if args.rows and args.cols:
        # result = cut_raster_file_to_smaller_boxes(args.inputfile, folders, args.rows, args.cols)
    # else:
        # result = cut_raster_file_to_smaller_boxes(args.inputfile, folders)

    # result = choose_reference_values(folders)

    # result = identify_number_of_optimum_clusters(folders)

    result = k_means_clustering(folders)

    result = polygonize_after_k_means(folders)

    result = max_p_algorithm(folders)

    result = max_p_algorithm_2(folders)

    print('----------------------------- END --------------------------------')

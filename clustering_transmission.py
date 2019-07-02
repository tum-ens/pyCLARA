import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
import logging
import datetime

from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString
from shapely.geometry import mapping
from shapely.ops import polygonize
from scipy.spatial import Voronoi

import pdb


def cluster_trans_file(trans_file_path, voronoi_file_path, no_of_clusters=28):
    """
    This function clusters the transmission network into a specified number of clusters.
    :param trans_file_path: The transmission network shapefile path.
    :param voronoi_file_path: The clipped voronoi polygons file path.
    :param no_of_clusters: The number of clusters that the map will be cut into.
    :return:
    """
    print('Reading the transmission file as a geoDataFrame.')
    logger.info('Reading the transmission file "' + str(trans_file_path) + '" as a geoDataFrame.')
    gdf_trans = gpd.read_file(trans_file_path)

    # Adding two new columns to gdf_trans.
    gdf_trans['point_1'] = 0
    gdf_trans['point_2'] = 0
    gdf_trans['point_1'] = gdf_trans['point_1'].astype(object)
    gdf_trans['point_2'] = gdf_trans['point_2'].astype(object)

    # Getting points from all the lines which will be used to form voronoi polygons.
    logger.info('Adding the points of each transmission line to point columns in the geoDataFrame of transmission '
                'lines.')
    print('Adding the points of each transmission line to point columns in the geoDataFrame of transmission lines.')
    count = 0

    for index, row in gdf_trans.iterrows():
        # print(row['geometry'])
        try:
            coord = row['geometry'].coords
            point_1 = coord[0]
            point_2 = coord[1]
            gdf_trans.at[index, 'point_1'] = Point(point_1[0], point_1[1])
            gdf_trans.at[index, 'point_2'] = Point(point_2[0], point_2[1])
        except NotImplementedError:
            count += 1

    # print(gdf_trans)

    if count:
        logger.error('MULTILINESTRING geometries found. Clean the transmission shapefile first. Exiting.')
        print('MULTILINESTRING geometries found. Clean the file first. Exiting.')
        return False

    print('MULTILINESTRING geometries = ' + str(count))

    # Declaring a common crs.
    crs = {'init': 'epsg:4326'}

    # Transmission lines that end up in the ocean need to be deleted.
    # Reading the map of Europe in order to see which lines terminate in the ocean.
    print('Removing the transmission lines that terminate in ocean.')
    gdf_eu = gpd.read_file('Europe_1node.shp')
    gdf_eu.crs = crs

    # Creating geoDataFrame of points that make up the transmission lines.
    logger.info('Creating geoDataFrame of points that make up the transmission lines.')
    df_points_1 = gdf_trans['ID'].copy()
    df_points_2 = gdf_trans['ID'].copy()

    geometry_1 = [x for x in gdf_trans['point_1']]
    geometry_2 = [x for x in gdf_trans['point_2']]

    gdf_points_1 = gpd.GeoDataFrame(df_points_1, crs=crs, geometry=geometry_1)
    gdf_points_2 = gpd.GeoDataFrame(df_points_2, crs=crs, geometry=geometry_2)

    # Combining the two GeoDataFrames of points in order to perform spatial join with voronoi polygons GeoDataFrame.
    gdf_points_comb_clean = gpd.GeoDataFrame(pd.concat([gdf_points_1, gdf_points_2], ignore_index=True), crs=crs)

    logger.info('Doing spatial join of geoDataFrame of points with geoDataFrame of Europe.')
    print('Doing spatial join between points and Europe.')
    gdf_sjoin_clean = gpd.sjoin(gdf_points_comb_clean, gdf_eu, how='inner', op='within')

    logger.info('Resetting the index of gdf_sjoin.')
    gdf_sjoin_clean.reset_index(drop=True, inplace=True)

    # Renaming the NAME_0 of polygons to polygon_id for clarity.
    gdf_sjoin_clean.rename(columns={'NAME_0': 'polygon_id'}, inplace=True)

    logger.info('Sorting the values of gdf_sjoin based on "ID" column.')
    gdf_sjoin_clean.sort_values('ID', inplace=True)

    # Creating new columns in gdf_trans for origin and destination voronoi
    gdf_trans['org_poly'] = 'ocean'
    gdf_trans['dest_poly'] = 'ocean'

    # Setting the index of gdf_trans based on the ID in order to assist in the assigning of org and dest polygons in the
    # for loop below.
    logger.info('Setting the "ID" column of gdf_trans as index.')
    gdf_trans.set_index('ID', inplace=True)

    # Setting first True because for each id, two columns need to be set.
    first = True

    # Iterating through gdf_sjoin in order to assign origin and destination polygons to lines.
    print('Assigning origin and destination polygons to transmission lines to determine if point is in ocean or on '
          'land.')
    logger.info('Assigning origin and destination polygons to transmission lines(gdf_trans) for EU or ocean.')
    for index, row in gdf_sjoin_clean.iterrows():
        if first is True:
            gdf_trans.at[row['ID'], 'org_poly'] = row['polygon_id']
            first = False
        else:
            gdf_trans.at[row['ID'], 'dest_poly'] = row['polygon_id']
            first = True

    # Resetting the gdf_trans index so that ID column comes back.
    logger.info('Resetting the index of gdf_trans.')
    gdf_trans.reset_index(inplace=True)

    # pdb.set_trace()

    # Delete the lines that have origin or destination polygon as 0(which means that the line terminates in the ocean).
    print('Deleting the transmission lines that terminate in ocean.')
    logger.info('Deleting the transmission lines that terminate in the ocean.')
    gdf_trans = gdf_trans[(gdf_trans['dest_poly'] != 'ocean') & (gdf_trans['org_poly'] != 'ocean')]
    gdf_trans.reset_index(inplace=True)

    # Dropping the columns org_poly and dest_poly from gdf_trans.
    gdf_trans.drop('org_poly', axis=1, inplace=True)
    gdf_trans.drop('dest_poly', axis=1, inplace=True)

    # pdb.set_trace()

    # # Building points list for voronoi polygons.
    # print('Creating points list for voronoi polygons.')
    # logger.info('Creating points list for voronoi polygons.')
    # points_list = list()
    # for index, row in gdf_trans.iterrows():
    #     coord = row['geometry'].coords
    #     point_1 = coord[0]
    #     point_2 = coord[1]
    #     points_list.append([point_1[0], point_1[1]])
    #     points_list.append([point_2[0], point_2[1]])
    #
    # # Creating voronoi polygons.
    # print('Creating voronoi polygons.')
    # logger.info('Creating voronoi polygons.')
    # voronoi_polygons = create_voronoi_polygons(points_list)
    #
    # # Define a polygon feature geometry with one attribute
    # schema = {
    #     'geometry': 'Polygon',
    #     'properties': {'id': 'int'},
    # }
    #
    # # Write a new shapefile.
    # voronoi_file = 'voronoi_file_latest.shp'
    # print('Saving file: ' + str(voronoi_file))
    # logger.info('Saving file: ' + str(voronoi_file))
    # # sjoin can be used to retain the CL_no attribute.
    # with fiona.open(voronoi_file, 'w', 'ESRI Shapefile', schema) as file:
    #     # If there are multiple geometries, put the "for" loop here
    #     i = 1
    #     for v in voronoi_polygons:
    #         file.write({
    #             'geometry': mapping(v),
    #             'properties': {'id': i},
    #         })
    #         i += 1
    #
    # pdb.set_trace()

    # MANUALLY CLIP BEFORE CONTINUING.
    logger.info('Reading the voronoi polygons shapefile as a geoDataFrame.')
    print('Reading the voronoi polygons shapefile as a geoDataFrame.')
    gdf_voronoi = gpd.read_file(voronoi_file_path)
    gdf_voronoi.crs = crs

    # Rebuilding gdf_points_comb so that the points from deleted lines are eliminated.
    # Creating geoDataFrame of points that make up the transmission lines.
    logger.info('Creating geoDataFrame of points that make up the transmission lines after filtering of lines that '
                'terminate in the ocean.')
    df_points_1 = gdf_trans['ID'].copy()
    df_points_2 = gdf_trans['ID'].copy()

    geometry_1 = [x for x in gdf_trans['point_1']]
    geometry_2 = [x for x in gdf_trans['point_2']]

    gdf_points_1 = gpd.GeoDataFrame(df_points_1, crs=crs, geometry=geometry_1)
    gdf_points_2 = gpd.GeoDataFrame(df_points_2, crs=crs, geometry=geometry_2)

    # Combining the two GeoDataFrames of points in order to perform spatial join with voronoi polygons GeoDataFrame.
    gdf_points_comb = gpd.GeoDataFrame(pd.concat([gdf_points_1, gdf_points_2], ignore_index=True), crs=crs)

    # pdb.set_trace()
    logger.info('Doing spatial join of geoDataFrame of points with geoDataFrame of voronoi polygons.')
    print('Doing spatial join between points and voronoi polygons.')
    gdf_sjoin = gpd.sjoin(gdf_points_comb, gdf_voronoi, how='inner', op='within')

    logger.info('Resetting the index of gdf_sjoin.')
    gdf_sjoin.reset_index(drop=True, inplace=True)

    # Renaming the id of polygons to polygon_id for clarity.
    gdf_sjoin.rename(columns={'id': 'polygon_id'}, inplace=True)

    logger.info('Sorting the values of gdf_sjoin based on "ID" column.')
    gdf_sjoin.sort_values('ID', inplace=True)

    # Creating new columns in gdf_trans for origin and destination voronoi
    gdf_trans['org_poly'] = 0
    gdf_trans['dest_poly'] = 0

    # Setting the index of gdf_trans based on the ID in order to assist in the assigning of org and dest polygons in the
    # for loop below.
    logger.info('Setting the "ID" column of gdf_trans as index.')
    gdf_trans.set_index('ID', inplace=True)

    # Setting first True because for each id, two columns need to be set.
    first = True

    # Iterating through gdf_sjoin in order to assign origin and destination polygons to lines.
    print('Assigning origin and destination voronoi polygons ID to transmission lines.')
    logger.info('Assigning origin and destination polygons to transmission lines(gdf_trans).')
    for index, row in gdf_sjoin.iterrows():
        if first is True:
            gdf_trans.at[row['ID'], 'org_poly'] = row['polygon_id']
            first = False
        else:
            gdf_trans.at[row['ID'], 'dest_poly'] = row['polygon_id']
            first = True

    # Resetting the gdf_trans index so that ID column comes back.
    logger.info('Resetting the index of gdf_trans.')
    gdf_trans.reset_index(inplace=True)

    # pdb.set_trace()

    # Delete the lines that have origin or destination polygon as 0(which means that the line terminates in the ocean).
    print('Deleting the transmission lines that still terminate in ocean.')
    logger.info('Deleting the transmission lines that still terminate in the ocean.')
    gdf_trans = gdf_trans[(gdf_trans['dest_poly'] != 0) & (gdf_trans['org_poly'] != 0)]

    # Initializing new columns in gdf_voronoi.
    logger.info('Adding new columns to gdf_voronoi and defining types.')
    gdf_voronoi['elec_neighbors'] = 0
    gdf_voronoi['trans_lines'] = 0
    gdf_voronoi['Area'] = 0
    gdf_voronoi['Cap'] = 0
    gdf_voronoi['Ratio'] = 0  # Ratio is Capacity/Area.

    # Setting the data types of the columns in gdf_voronoi. It converts the GeoDataFrame object to a DataFrame object.
    gdf_voronoi = gdf_voronoi.astype({'elec_neighbors': object, 'trans_lines': object, 'Cap': float, 'Ratio': float,
                                      'Area': float})

    # Now gdf_voronoi is DataFrame object and it needs to be converted back to a GeoDataFrame.
    logger.info('Due to a bug in geopandas, setting the type of geoDataFrame changes it to DataFrame. Setting it back'
                ' to a geoDataFrame.')
    gdf_voronoi = gpd.GeoDataFrame(gdf_voronoi, crs=crs, geometry='geometry')

    # Calculating the area of the polygons in gdf_voronoi.
    logger.info('Changing the crs of gdf_voronoi to epsg:3034 in order to calculate area of voronoi polygons.')
    print('Changing gdf_voronoi crs to epsg:3034.')
    crs_area = {'init': 'epsg:3034'}
    gdf_voronoi = gdf_voronoi.to_crs(crs_area)
    print('Calculating area of voronoi polygons.')
    logger.info('Calculating the area of voronoi polygons')
    gdf_voronoi['Area'] = gdf_voronoi.area / 10**6

    # Iterating through gdf_lines in order to build the gdf_voronoi according to needs.
    print('Adding electric neighbors and other attributes to gdf_voronoi.')
    logger.info('Adding electric neighbors, trans_lines, Area, Capacity and Ratio to gdf_voronoi.')
    poly_without_neighbor = 0
    for index, row in gdf_voronoi.iterrows():
        # Getting the electric neighbors of the polygons.
        neighbors_1 = gdf_trans['dest_poly'][gdf_trans['org_poly'] == row['id']].values
        neighbors_2 = gdf_trans['org_poly'][gdf_trans['dest_poly'] == row['id']].values
        all_neighbors = np.concatenate((neighbors_1, neighbors_2)).tolist()
        all_neighbors = list(set(all_neighbors))
        # This if statement ensures that 0 is not wrongly added as a neighbor for lines which have been clipped and do
        # not terminate properly.
        # if 0 in all_neighbors: NOT NEEDED AS THESE LINES HAVE BEEN DELETED
        #     all_neighbors.remove(0)
        if not all_neighbors:
            logger.debug('No electric neighbors found for voronoi polygon id: ' + str(row['id']))
            print('No electric neighbors found for voronoi polygon id: ' + str(row['id']))
            print('Deleting the voronoi polygon.')
            gdf_voronoi.drop(index, inplace=True)
            poly_without_neighbor += 1
            continue
        else:
            gdf_voronoi.at[index, 'elec_neighbors'] = all_neighbors

        # Getting the list of transmission lines that connect the polygon.
        trans_lines_1 = gdf_trans['ID'][gdf_trans['org_poly'] == row['id']].values
        trans_lines_2 = gdf_trans['ID'][gdf_trans['dest_poly'] == row['id']].values
        all_trans_lines = np.concatenate((trans_lines_1, trans_lines_2)).tolist()
        all_trans_lines = list(set(all_trans_lines))
        if not all_trans_lines:
            logger.debug('No transmission line found for voronoi id: ' + str(row['id']))
            print('No transmission line for voronoi id: ' + str(row['id']))
            gdf_voronoi.at[index, 'trans_lines'] = 0
            pdb.set_trace()
        else:
            gdf_voronoi.at[index, 'trans_lines'] = all_trans_lines

        # Sum the capacities of all the lines here.
        cap_1 = gdf_trans['Cap_MVA'][gdf_trans['org_poly'] == row['id']].values
        cap_2 = gdf_trans['Cap_MVA'][gdf_trans['dest_poly'] == row['id']].values
        all_cap = np.concatenate((cap_1, cap_2))
        sum_cap = sum(all_cap)
        gdf_voronoi.at[index, 'Cap'] = sum_cap

        # Calculating the ratio
        gdf_voronoi.at[index, 'Ratio'] = sum_cap / row['Area']

    print('Total voronoi polygons with 0 electric neighbors = ' + str(poly_without_neighbor))

    pdb.set_trace()

    print('Resetting the index of gdf_voronoi.')
    logger.info('Resetting the index of gdf_voronoi.')
    gdf_voronoi.reset_index(inplace=True, drop=True)

    islands = 0

    len_gdf_voronoi = len(gdf_voronoi.index)
    # Doing a while loop on gdf_voronoi in order to reduce the total number of polygons to the desired number of
    # clusters.
    logger.info('Dissolving polygons in gdf_voronoi based on the highest ratio')
    print('Dissolving polygons based on the highest ratio.')
    time_current = datetime.datetime.now()
    time_formatted = time_current.strftime("%Y-%m-%d_%H:%M:%S")
    logger.info('Current time: ' + time_formatted)
    logger.info('Len:' + str(len_gdf_voronoi))
    print('Current time: ' + time_formatted)
    print('Len:' + str(len_gdf_voronoi))
    while len_gdf_voronoi > max(islands, no_of_clusters):
        # Getting the index of the polygon with the highest ratio.
        poly1_index = gdf_voronoi['Ratio'].idxmax()
        poly1 = gdf_voronoi.loc[poly1_index]
        # print(poly1['id'])

        # Getting the list of electric neighbors of poly1.
        elec_neighbors = poly1['elec_neighbors']

        try:
            if len(elec_neighbors) == 0:
                pdb.set_trace()
        except TypeError:
            logger.exception(TypeError)
            logger.exception('Cannot calculate length of elec_neighbors.')
            pdb.set_trace()

        # Getting the dataframe of electric neighbors.
        poly1_neighbors = gdf_voronoi[gdf_voronoi['id'].isin(elec_neighbors)]

        # Getting the index of polygon from electric neighbors which has the highest ratio. It is named poly2.
        poly2_index = poly1_neighbors['Ratio'].idxmax()
        poly2 = poly1_neighbors.loc[poly2_index]

        # Converting poly1 and poly2 Series object to geoDataFrame object.
        poly1 = poly1.to_frame()
        poly1 = poly1.transpose()
        poly1 = gpd.GeoDataFrame(poly1, geometry='geometry', crs=crs_area)
        poly2 = poly2.to_frame()
        poly2 = poly2.transpose()
        poly2 = gpd.GeoDataFrame(poly2, geometry='geometry', crs=crs_area)

        # Adding buffer(0) to poly1_gdf and poly2_gdf in order to avoid self-intersections.
        poly1['geometry'] = poly1.geometry.buffer(0)
        poly2['geometry'] = poly2.geometry.buffer(0)

        # Setting Ratio of poly1 and poly2 to 0 so that they can be dissolved.
        poly1['Ratio'] = 0
        poly2['Ratio'] = 0

        # print('ID\'s for dissolve. POLY1 = ' + str(poly1['id'].values[0]) + ' and POLY2 = ' + str(poly2['id'].values[0]))

        # Combining poly1 and poly2 to form a GeoDataFrame.
        poly_comb = gpd.GeoDataFrame(pd.concat([poly1, poly2], ignore_index=True), crs=crs_area)
        # print(poly_comb)

        # Dissolving the two polygons and giving a new index and id.
        try:
            poly_dissolved = poly_comb.dissolve(by=['Ratio'])
            poly_dissolved.reset_index(inplace=True)
            poly_dissolved.id = [gdf_voronoi.id.max() + 1]
        except Exception as e:
            logger.exception(e)
            print(e)
            pdb.set_trace()
        # print('id', poly_dissolved.id.values[0])

        # Getting the new trans_lines, elec_neighbors, Area and Capacity for the newly formed polygon.
        # aggfunc in dissolve function can be used but some problems.
        poly_dissolved['Area'] = float(poly1['Area']) + float(poly2['Area'])
        poly_dissolved['Cap'] = float(poly1['Cap']) + float(poly2['Cap'])

        poly1_n = poly1['elec_neighbors'].values[0]
        poly2_n = poly2['elec_neighbors'].values[0]

        # Removing poly1['id'] from poly2['elec_neighbors'] and vice versa.
        poly1_id = poly1['id'].values[0]
        poly2_id = poly2['id'].values[0]
        poly1_n.remove(poly2_id)
        poly2_n.remove(poly1_id)
        poly_dissolved.at[0, 'elec_neighbors'] = list(set(poly1_n + poly2_n))  # set is used to remove duplicates.

        # Find common neighbors between poly1 and poly2 and just let it remain in one list (either poly1_n or poly2_n).
        # This will eliminate the bug that is introduced while assigning and correcting neighbors in for loops below.
        # common_n = list(set(poly1_n).intersection(poly2_n))

        # Also need to change elec_neighbors column of all associated polygons. ->DONE
        # The new id of the poly_dissolved also needs to be added to elec_neighbors of all others. ->DONE
        for poly_id in poly1_n:
            gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0].remove(poly1_id)
            if poly_dissolved['id'].values[0] not in gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0]:
                gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0].append(poly_dissolved['id'].values[0])

        for poly_id in poly2_n:
            gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0].remove(poly2_id)
            if poly_dissolved['id'].values[0] not in gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0]:
                gdf_voronoi[gdf_voronoi['id'] == poly_id]['elec_neighbors'].values[0].append(poly_dissolved['id'].values[0])

        # Only 1 common transmission line between two polygons being removed. Finding the common transmission line.
        # !!!This is not necessary. There can be more than one common_trans_line between two neighbors after some polygons
        # have been dissolved. THIS IS PROBABLY WHERE THE BUG IS INTRODUCED!!!
        poly1_trans_lines = poly1['trans_lines'].values[0]
        poly2_trans_lines = poly2['trans_lines'].values[0]
        common_trans_line = list(set(poly1_trans_lines).intersection(poly2_trans_lines))

        cap_all_common_lines = 0.0
        for line in common_trans_line:
            poly1_trans_lines.remove(line)
            poly2_trans_lines.remove(line)
            cap_all_common_lines += float(gdf_trans[gdf_trans['ID'] == line]['Cap_MVA'].values[0])

        # poly1_trans_lines.remove(common_trans_line)
        # poly2_trans_lines.remove(common_trans_line)
        poly_dissolved.at[0, 'trans_lines'] = list(poly1_trans_lines + poly2_trans_lines)

        # From the total capacity of poly1 and poly2, the capacity of common_trans_line must be subtracted in order to
        # get the correct capacity.
        poly_dissolved['Cap'] = poly_dissolved['Cap'] - cap_all_common_lines

        # Calculating new ratio for poly_dissolved.
        poly_dissolved['Ratio'] = poly_dissolved['Cap'] / poly_dissolved['Area']

        # Deleting poly1 and poly2 from gdf_voronoi.
        gdf_voronoi = gdf_voronoi.drop([poly1_index, poly2_index])

        # Checking if 0 electric neighbors of poly dissolved, then setting the ratio to 0.
        poly_dissolved_elec_neighbors = poly_dissolved['elec_neighbors'].values[0]
        if len(poly_dissolved_elec_neighbors) == 0:
            islands += 1
            print('No electric neighbors. No. of islands:' + str(islands))
            poly_dissolved['Ratio'] = 0
            # pdb.set_trace()

        # Adding poly_dissolved to gdf_voronoi.
        gdf_voronoi = gdf_voronoi.append(poly_dissolved, ignore_index=True)

        # Updating len_gdf_voronoi.
        try:
            len_gdf_voronoi = len(gdf_voronoi.index)
        except TypeError:
            pdb.set_trace()

        if len(gdf_voronoi) % 100 == 0:
            time_current = datetime.datetime.now()
            time_formatted = time_current.strftime("%Y-%m-%d_%H:%M:%S")
            logger.info('Current time: ' + time_formatted)
            logger.info('Len:' + str(len_gdf_voronoi))
            print('Current time: ' + time_formatted)
            print('Len:' + str(len_gdf_voronoi))

        if len(gdf_voronoi) == 4000:
            gdf_voronoi.to_file(driver='ESRI Shapefile', filename='final_trans_result_4000.shp')
            pdb.set_trace()

    pdb.set_trace()
    try:
        # Dropping elec_neighbors and trans_lines columns because list stored in object data type in geoDataFrame throws
        # an error.
        logger.info('Dropping columns "elec_neighbors" and "trans_lines" from gdf_voronoi.')
        gdf_voronoi.drop('elec_neighbors', axis=1, inplace=True)
        gdf_voronoi.drop('trans_lines', axis=1, inplace=True)
        logger.info('Saving final shapefile "final_trans_result.shp".')
        gdf_voronoi.to_file(driver='ESRI Shapefile', filename='final_trans_result.shp')
    except Exception as e:
        print(e)
        pdb.set_trace()


def create_voronoi_polygons(points_list):
    """
    This function makes voronoi polygons by taking a points list as input.
    :param points_list: The points list is used to make voronoi polygons.
    :return:
    """

    voronoi_polygons = Voronoi(points_list)

    # Make lines from voronoi polygons.
    lines = [LineString(voronoi_polygons.vertices[line])
             for line in voronoi_polygons.ridge_vertices
             ]

    # Return list of polygons created from lines.
    polygons_list = list()  # polygons

    for polygon in polygonize(lines):
        polygons_list.append(polygon)

    # pdb.set_trace()

    return polygons_list


if __name__ == '__main__':
    #  Setting basic config for logger.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='log_connecting_islands_2.txt')  # pass explicit filename here
    logger = logging.getLogger()
    print('------------------------- Starting Clustering -------------------------')
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('Started at: ' + formatted_time)

    clean_clipped_trans_file = 'clipped_gridkit_connected.shp'
    voronoi_file = 'clipped_voronoi_latest.shp'
    cluster_trans_file(clean_clipped_trans_file, voronoi_file)

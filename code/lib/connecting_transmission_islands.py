"""
This script takes a shape file of transmission network and uses the graph theory to identify the islands.
After the identification of islands, it makes connections between those islands.
To run this script, specify the name of the transmission shape file in the trans_file variable.
Output will be a shapefile called "trans_connected.shp".
Sample command to run the script: python connecting_transmission_islands.py

Packages required are as follows:
1. networkx
2. pandas
3. geopandas
4. numpy
5. shapely
6. scipy
"""

import pdb
import logging
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString
from scipy.spatial import cKDTree


def connect_islands(trans_file_path):
    """
    This function connects islands of transmission network.
    :param trans_file_path: GeoDataFrame of the transmission network.
    :return: Shapefile with connections between islands.
    """
    # Reading transmission file as geoDataFrame.
    logger.info('Reading file "' + str(trans_file_path) + '" as geoDataFrame.')
    print('Reading shapefile as a geoDataFrame.')
    gdf_trans = gpd.read_file(trans_file_path)

    # Reading the shapefile as a networkx graph.
    logger.info('Reading file "' + str(trans_file_path) + '" as networkx graph.')
    print('Reading shapefile as a graph.')
    graph = nx.read_shp('clipped_gridkit_clean.shp')

    # Converting graph to undirected graph.
    graph = graph.to_undirected()

    # Getting islands in the graph as sub-graphs.
    print('Getting electric islands in transmission network.')
    islands_graph = list(nx.connected_component_subgraphs(graph))
    print('No. of islands: ' + str(len(islands_graph)))
    logger.info('No of islands: ' + str(len(islands_graph)))

    # Setting the crs.
    crs = {'init': 'epsg:4326'}

    gdfs_dict = dict()

    # Making each graph island a GeoDataFrame.
    print('Converting graphs to GeoDataFrames of points.')
    logger.info('Converting graphs to GeoDataFrames of points.')
    for index, island in enumerate(islands_graph):
        nodes = list(island.nodes())
        points = list()
        for node in nodes:
            points.append(Point(node[0], node[1]))

        df = pd.DataFrame(points, columns=['geometry'])

        gdfs_dict[index] = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

    print('Finding closest point to other islands for all islands and connecting them.')
    logger.info('Finding closest point to other islands for all islands and connecting them.')

    # All new lines will have a capacity of 100 MVA.
    cap_mva = 100.0

    # This while loop will execute until only 1 island remains.
    while len(gdfs_dict) != 1:
        print('No of islands: ' + str(len(gdfs_dict)))
        key_1 = 0
        val_1 = gdfs_dict[key_1]
        results = pd.DataFrame(columns=['gdf', 'index', 'distance', 'point'])
        for key_2, val_2 in gdfs_dict.items():
            # if-else block makes sure that the same gdf is not compared when calculating distances to nearest nodes.
            if key_1 == key_2:
                continue
            else:
                df = ckd_nearest(val_1, val_2, 'geometry')

                # Getting the index of the point that has the minimum distance.
                idx = df[['distance']].idxmin().values[0]

                # Getting the distance and the point that is nearest.
                dist = df.loc[idx].values[0]
                pt = df.loc[idx].values[1]
                row = [key_2, idx, dist, pt]

                # Adding row to the dataframe.
                results.loc[len(results)] = row

        # Finding the gdf whose point has the closest distance to gdf[key1].
        # Getting the index of the gdf results for which there is minimum distance.
        idx_min = results[['distance']].idxmin().values[0]

        # Getting the row from results according to idx_min.
        rq_row = results.loc[idx_min]

        # Getting the point for gdf_1 with the help of 'index' column in rq_row.
        point_1 = gdfs_dict[key_1].loc[rq_row['index']].values[0]

        # gdf_1 is key_1. Getting gdf_2 from rq_row with the help of 'gdf' column.
        gdf_2 = rq_row['gdf']

        # Getting point_2 from rq_row with 'point' column in rq_row.
        point_2 = rq_row['point']

        # Make a LINESTRING between point_1 and point_2 in gdf_trans.
        id_use = gdf_trans['ID'].max() + 1
        insert_row = [id_use, cap_mva, LineString([point_1, point_2])]
        gdf_trans.loc[len(gdf_trans)] = insert_row

        # Append gdf_2 to gdf_1 and delete gdf_2.
        gdfs_dict[key_1] = gdfs_dict[key_1].append(gdfs_dict[gdf_2], ignore_index=True)
        del gdfs_dict[gdf_2]

    # pdb.set_trace()

    # Writing the connected GeoDataFrame(gdf_trans) to a new shapefile.
    print('Writing connected shapefile: "trans_connected.shp".')
    logger.info('Writing connected shapefile: "trans_connected.shp".')
    gdf_trans.to_file(driver='ESRI Shapefile', filename='trans_connected.shp')

    return True


def ckd_nearest(gdf_a, gdf_b, bcol):
    """
    This function finds the distance and the nearest points in gdf_b for every point in gdf_a.
    :param gdf_a: GeoDataFrame of Points.
    :param gdf_b: GeoDataFrame of Points.
    :param bcol: Column that should be listed in the resulting DataFrame.
    :return:
    """
    na = np.array(list(zip(gdf_a.geometry.x, gdf_a.geometry.y)))
    nb = np.array(list(zip(gdf_b.geometry.x, gdf_b.geometry.y)))
    btree = cKDTree(nb)
    dist, idx = btree.query(na, k=1)
    df = pd.DataFrame.from_dict({'distance': dist.astype(float), 'bcol': gdf_b.loc[idx, bcol].values})

    return df


def clean_clipped_trans_file(clipped_file, original_file):
    """
    This function cleans the clipped transmission file by replacing the MULTILINESTRING instances with LINESTRING
    instances. MULTILINESTRING instances are formed as a result of clipping.
    :param clipped_file:  Gridkit transmission file clipped with Europe map.
    :param original_file: Original Gridkit file.
    :return:
    """
    # Reading the transmission file.
    gdf_trans = gpd.read_file(clipped_file)

    # Reading the gridkit_cleaned file. This is done in order to replace the MULTILINESTRING geometries that are formed
    # as a result of clipping.
    gdf_clean_trans = gpd.read_file(original_file)

    # Replacing all lines in clipped transmission lines file with lines from clean transmission file so that
    # clipped transmission lines(now MULTILINESTRINGs) get converted to LINESTRING objects.
    for index, row in gdf_trans.iterrows():
        id_feature = row['ID']
        one_feature_clean_trans = gdf_clean_trans[gdf_clean_trans['ID'] == id_feature]
        gdf_trans.at[index, 'geometry'] = one_feature_clean_trans['geometry'].iloc[0]

    gdf_trans.to_file(driver='ESRI Shapefile', filename='trans_clean.shp')


if __name__ == '__main__':
    #  Setting basic config for logger.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='log_connecting_islands.txt')  # pass explicit filename here
    logger = logging.getLogger()
    trans_file = 'clipped_gridkit_clean.shp'
    connect_islands(trans_file)

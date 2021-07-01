from .spatial_functions import ckd_nearest, assign_disconnected_components_to_nearest_neighbor
from .util import *


def lines_clustering(paths, param):
    """
    This function applies the hierarchical clustering algorithm to the shapefile of transmission lines. It first ensures that the whole
    grid is one component (no electric islands), by eventually adding fake lines with low capacity. Then it clips the grid to the scope, and
    creates a shapefile of voronoi polygons based on the points at the start/end of the lines. Regions with a small area and high connectivity
    to their neighbors are aggregated together, until the target number of regions is reached.
    
    :param paths: Dictionary of paths pointing to input files and output locations.
    :type paths: dict
    :param param: Dictionary of parameters including transmission-line-related parameters.
    :type param: dict
    
    :return: The called functions :mod:`connect_islands`, :mod:`clip_transmission_shapefile`, :mod:`create_voronoi_polygons`, and :mod:`cluster_transmission_shapefile` generate outputs.
    :rtype: None
    """
    timecheck("Start")

    connect_islands(paths, param)
    clip_transmission_shapefile(paths, param)
    create_voronoi_polygons(paths, param)
    cluster_transmission_shapefile(paths, param)

    timecheck("End")


def connect_islands(paths, param):
    """
    This script takes a shapefile of a transmission network and uses graph theory to identify its components (electric islands).
    After the identification of islands, it creates connections between them using additional transmission lines with low capacities.
    This correction assumes that there are actually no electric islands, and that multiple graph components only exist because some
    transmission lines are missing in the data. The output is a shapefile of transmission lines with no electric islands.

    :param paths: Dictionary of paths including the path to the input shapefile of the transmission network, *grid_input*.
    :type paths: dict
    :param param: Dictionary of parameters including *CRS_grid*, *default_cap_MVA*, and *default_line_type*.
    :type paths: dict

    :return: The shapefile with connections between islands is saved directly in the desired path *grid_connected*.
    :rtype: None
    """
    timecheck("Start")

    # Reading transmission file as geoDataFrame
    gdf_trans = gpd.read_file(paths["grid_input"])

    # Reading the shapefile as a networkx undirected graph
    graph = nx.read_shp(paths["grid_input"]).to_undirected()

    # Getting islands in the graph as sub-graphs
    islands_graph = [graph.subgraph(c) for c in nx.connected_components(graph)]
    print("Number of electric islands: " + str(len(islands_graph)))

    # Making each graph island a GeoDataFrame
    gdfs_dict = dict()
    print("Converting graphs to GeoDataFrames of points.")
    for index, island in enumerate(islands_graph):
        nodes = list(island.nodes())
        points = list()
        for node in nodes:
            points.append(Point(node[0], node[1]))
        df = pd.DataFrame(points, columns=["geometry"])
        gdfs_dict[index] = gpd.GeoDataFrame(df, geometry="geometry", crs={"init": param["CRS_grid"]})

    # This while loop will execute until only one component remains.
    print("Finding closest point to other islands for all islands and connecting them.")
    while len(gdfs_dict) != 1:
        key_1 = 0
        val_1 = gdfs_dict[key_1]
        results = pd.DataFrame(columns=["gdf", "index", "distance", "point"])
        for key_2, val_2 in gdfs_dict.items():
            if key_1 != key_2:
                df = ckd_nearest(val_1, val_2, "geometry")

                # Get the index of the point that has the minimum distance
                idx = df[["distance"]].idxmin().values[0]

                # Get the distance and the point that is nearest
                dist = df.loc[idx].values[0]
                pt = df.loc[idx].values[1]
                row = [key_2, idx, dist, pt]

                # Add row to the dataframe
                results.loc[len(results)] = row

        # Find the gdf whose point has the closest distance to gdf[key1]
        # Get the index of the gdf results for which there is minimum distance
        idx_min = results[["distance"]].idxmin().values[0]

        # Get the row from results according to idx_min
        rq_row = results.loc[idx_min]

        # Get the point for gdf_1 with the help of 'index' column in rq_row
        point_1 = gdfs_dict[key_1].loc[rq_row["index"]].values[0]

        # gdf_1 is key_1. Getting gdf_2 from rq_row with the help of 'gdf' column
        gdf_2 = rq_row["gdf"]

        # Getting point_2 from rq_row with 'point' column in rq_row
        point_2 = rq_row["point"]

        # Make a LINESTRING between point_1 and point_2 in gdf_trans
        id_use = gdf_trans["ID"].max() + 1
        insert_row = [id_use, param["default_cap_MVA"], param["default_line_type"], LineString([point_1, point_2])]
        gdf_trans.loc[len(gdf_trans)] = insert_row

        # Append gdf_2 to gdf_1 and delete gdf_2
        gdfs_dict[key_1] = gdfs_dict[key_1].append(gdfs_dict[gdf_2], ignore_index=True)
        del gdfs_dict[gdf_2]
        print("Number of electric islands: " + str(len(gdfs_dict)))

    # Write the connected GeoDataFrame(gdf_trans) to a new shapefile
    print('Writing connected shapefile: paths["grid_connected"].')
    gdf_trans.crs = {"init": param["CRS_grid"]}
    gdf_trans.to_file(driver="ESRI Shapefile", filename=paths["grid_connected"])

    timecheck("End")


def clip_transmission_shapefile(paths, param):
    """
    This function clips the shapefile of the transmission lines using the shapefile of the scope.
    MULTILINESTRING instances are formed as a result of clipping. Hence, the script cleans the clipped transmission file by replacing the MULTILINESTRING instances with LINESTRING
    instances. 

    :param paths: Dictionary of paths including the path to the shapefile of the transmission network after connecting islands, *grid_connected*, and to the output *grid_clipped*.
    :type paths: dict
    :param param: Dictionary of parameters including *CRS_grid*.
    :type paths: dict

    :return: The shapefile of clipped transmission lines is saved directly in the desired path *grid_clipped*.
    :rtype: None
    """

    timecheck("Start")

    # Read the transmission file
    gdf_trans = gpd.read_file(paths["grid_connected"])

    # Clip the transmission file
    gdf_scope = gpd.read_file(paths["spatial_scope"])
    gdf_scope = gdf_scope.to_crs(param["CRS_grid"])
    gdf_clipped = gpd.clip(gdf_trans, gdf_scope, keep_geom_type=True)

    # Replacing all lines in clipped transmission lines file with lines from original transmission file so that
    # clipped transmission lines (now MULTILINESTRINGs) get converted to LINESTRING objects.
    gdf_clipped_filtered = gdf_trans.loc[gdf_clipped.index]

    # Save output
    gdf_clipped_filtered.to_file(driver="ESRI Shapefile", filename=paths["grid_clipped"])

    timecheck("End")


def create_voronoi_polygons(paths, param):
    """
    This function creates a shapefile of voronoi polygons based on the points at the start/end of the lines.
    
    :param paths: Dictionary of paths including the path to the shapefile of the transmission network after clipping, *grid_clipped*, to the scope *spatial_scope*, and to the output *grid_voronoi*.
    :type paths: dict
    :param param: Dictionary of parameters including *CRS_grid*.
    :type paths: dict

    :return: The shapefile of voronoi polygons is saved directly in the desired path *grid_voronoi*.
    :rtype: None
    """

    # Read the transmission file
    gdf_trans = gpd.read_file(paths["grid_clipped"])

    # Get list of points
    points_list = []
    for ind in gdf_trans.index:
        for y in gdf_trans.loc[ind, "geometry"].coords:
            points_list.append(y)
    points_list = list(set(points_list))

    # Read shapefile of scope
    gdf_scope = gpd.read_file(paths["spatial_scope"])
    gdf_scope = gdf_scope.to_crs(param["CRS_grid"])

    # Get radius for voronoi polygons (empirical value)
    scope_bounds = gdf_scope.total_bounds
    rad = max(scope_bounds[2] - scope_bounds[0], scope_bounds[3] - scope_bounds[1]) / 8

    # Create polygons
    regions, coordinates = voronoi(points_list, radius=rad)
    voronoi_polygons = [Polygon([coordinates[x] for x in regions[y]]) for y in range(len(regions))]
    df_voronoi = pd.DataFrame(voronoi_polygons).rename(columns={0: "geometry"})
    gdf_voronoi = gpd.GeoDataFrame(df_voronoi, geometry="geometry", crs=param["CRS_grid"])

    # Convert scope into one single polygon
    poly = pd.DataFrame(columns=["NAME_SHORT", "geometry"])
    poly.loc[0] = ("Scope", gdf_scope.geometry.unary_union)
    scope = gpd.GeoDataFrame(poly, geometry="geometry", crs=param["CRS_grid"])

    # Clip based on scope
    gdf_result = gpd.overlay(gdf_voronoi, scope, how="intersection", make_valid=True, keep_geom_type=True)

    # Edit columns
    gdf_result["ID_polygon"] = gdf_result.index
    gdf_result.drop(columns=["NAME_SHORT"], inplace=True)

    # Save output
    gdf_result.to_file(driver="ESRI Shapefile", filename=paths["grid_voronoi"])


def update_values_in_geodataframes(gdf_trans, gdf_voronoi, poly1, poly2, cluster_no):
    """
    This function updates the values in the geodataframes *gdf_trans* and *gdf_voronoi* after dissolving the polygons and is called in the loops in :mod:`cluster_transmission_shapefile`.
    
    :param gdf_trans: Geodataframe of transmission lines, containing the columns *Start_poly* and *End_poly*.
    :type gdf_trans: geopandas GeoDataFrame
    :param gdf_voronoi: Geodataframe of polygons, containing the columns *trans_lines*, *elec_neighbors*, *Cluster*, *Cap*, *Area*, and *Ratio*.
    :type gdf_voronoi: geopandas GeoDataFrame
    :param poly1: First polygon to be dissolved, containing the same columns as gdf_voronoi.
    :type poly1: geopandas GeoDataFrame
    :param poly2: Second polygon to be dissolved, containing the same columns as gdf_voronoi.
    :type poly2: geopandas GeoDataFrame
    :param cluster_no: Cluster number to be used for the dissolved polygons.
    :type cluster_no: integer
    
    :return (gdf_trans, gdf_voronoi): Updated geodataframes after dissolving poly1 and poly2.
    :rtype: tuple of geodataframes
    """
    # Update values in gdf_trans
    gdf_trans.loc[(gdf_trans["Start_poly"] == poly1["Cluster"]) | (gdf_trans["Start_poly"] == poly2["Cluster"]), "Start_poly"] = cluster_no
    gdf_trans.loc[(gdf_trans["End_poly"] == poly1["Cluster"]) | (gdf_trans["End_poly"] == poly2["Cluster"]), "End_poly"] = cluster_no
    gdf_trans = gdf_trans.loc[gdf_trans["Start_poly"] != gdf_trans["End_poly"]]

    # Update values in gdf_voronoi
    for ind in set([cluster_no]).union(poly1["elec_neighbors"]).union(poly2["elec_neighbors"]):
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == ind, "trans_lines"] = [
            set(gdf_trans.loc[gdf_trans["Start_poly"] == ind].index) | set(gdf_trans.loc[gdf_trans["End_poly"] == ind].index)
        ]
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == ind, "elec_neighbors"] = [
            set(gdf_trans.loc[gdf_trans["Start_poly"] == ind, "End_poly"]) | set(gdf_trans.loc[gdf_trans["End_poly"] == ind, "Start_poly"])
        ]
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == ind, "Cap"] = gdf_trans.loc[
            (gdf_trans["Start_poly"] == ind) | (gdf_trans["End_poly"] == ind), "Cap_MVA"
        ].sum()
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == ind, "Area"] = gdf_voronoi.loc[gdf_voronoi["Cluster"] == ind, "Area"].sum()
    gdf_voronoi["Ratio"] = gdf_voronoi["Cap"] / gdf_voronoi["Area"]
    return gdf_trans, gdf_voronoi


def cluster_transmission_shapefile(paths, param):
    """
    This function clusters the transmission network into a specified number of clusters.
    It first reads the shapefile of voronoi polygons, and initializes its attributes *elec_neighbors*, *trans_lines*,
    *Area*, *Cap*, and *Ratio*. Starting with the polygon with the highest ratio, it merges it with its electric neighbors
    with the highest ratio as well. It then updates the values and repeats the algorithm, until the target number of
    clusters is reached.
    
    :param paths: Dictionary of paths pointing to *grid_clipped*, *grid_debugging*, *grid_voronoi*, and to the outputs *grid_intermediate* and *grid_regions*.
    :type paths: dict
    :param param: Dictionary containing the parameter *CRS_grid* and the user preferences *number_clusters* and *intermediate_number*.
    :type param: dict
    
    :return: The intermediate and final outputs are saved directly in the desired shapefiles.
    :rtype: None
    """
    timecheck("Start")

    # Read the transmission file
    gdf_trans = gpd.read_file(paths["grid_clipped"])
    gdf_trans.crs = {"init": param["CRS_grid"]}

    # Read shapefile of voronoi polygons
    if os.path.exists(paths["grid_debugging"]):
        gdf_voronoi = gpd.read_file(paths["grid_debugging"])
        gdf_voronoi.index = gdf_voronoi["ID_polygon"]
    else:
        gdf_voronoi = gpd.read_file(paths["grid_voronoi"])

        # Initialize new columns
        gdf_voronoi["elec_neighbors"] = 0  # Will be a list of electrical neighbors (polygons connected through a line)
        gdf_voronoi["trans_lines"] = 0  # Will be a list of transmission lines going through the polygon
        gdf_voronoi["Area"] = 0  # Area of the polygon
        gdf_voronoi["Cap"] = 0  # Total capacity of lines going through the polygon
        gdf_voronoi["Ratio"] = 0  # Ratio is Capacity/Area

        # Setting the data types of the columns in gdf_voronoi. It converts the GeoDataFrame object to a DataFrame object
        gdf_voronoi = gdf_voronoi.astype({"elec_neighbors": object, "trans_lines": object, "Cap": float, "Ratio": float, "Area": float})

        # Now gdf_voronoi is DataFrame object and it needs to be converted back to a GeoDataFrame
        gdf_voronoi = gpd.GeoDataFrame(gdf_voronoi, crs=param["CRS_grid"], geometry="geometry")

        # Calculating the area of each polygon using Lambert Cylindrical Equal Area EPSG:9835
        if param["CRS_grid"] == "epsg:4326":
            gdf_voronoi = gdf_voronoi.to_crs("+proj=cea")
            gdf_voronoi["Area"] = gdf_voronoi["geometry"].area / 10 ** 6
            gdf_voronoi = gdf_voronoi.to_crs(epsg=4326)
        else:
            gdf_voronoi["Area"] = gdf_voronoi["geometry"].area / 10 ** 6

    # Calculate total area of scope (used as a constraint for the aggregation)
    total_area = gdf_voronoi["Area"].sum()

    # Find start and end points of the lines
    gdf_trans["Start_point"] = Point(gdf_trans.iloc[0]["geometry"].coords[0])
    gdf_trans["End_point"] = Point(gdf_trans.iloc[0]["geometry"].coords[1])
    for ind in gdf_trans.index:
        gdf_trans.loc[ind, "Start_point"] = Point(gdf_trans.loc[ind, "geometry"].coords[0])
        gdf_trans.loc[ind, "End_point"] = Point(gdf_trans.loc[ind, "geometry"].coords[1])

    # Find start and end polygons
    gdf_trans = gpd.GeoDataFrame(gdf_trans, geometry="Start_point", crs=param["CRS_grid"])
    gdf_trans = gpd.sjoin(gdf_trans, gdf_voronoi[["geometry"]], how="inner", op="within").rename(columns={"index_right": "Start_poly"})
    gdf_trans = gpd.GeoDataFrame(gdf_trans, geometry="End_point", crs=param["CRS_grid"])
    gdf_trans = gpd.sjoin(gdf_trans, gdf_voronoi[["geometry"]], how="inner", op="within").rename(columns={"index_right": "End_poly"})
    gdf_trans = gpd.GeoDataFrame(gdf_trans, geometry="geometry", crs=param["CRS_grid"])
    gdf_trans = gdf_trans.loc[gdf_trans["Start_poly"] != gdf_trans["End_poly"]]

    # Update values in gdf_voronoi
    for ind in gdf_voronoi.index:
        gdf_voronoi.loc[ind, "trans_lines"] = [
            set(gdf_trans.loc[gdf_trans["Start_poly"] == ind, "ID"]) | set(gdf_trans.loc[gdf_trans["End_poly"] == ind, "ID"])
        ]
        gdf_voronoi.loc[ind, "elec_neighbors"] = [
            set(gdf_trans.loc[gdf_trans["Start_poly"] == ind, "End_poly"]) | set(gdf_trans.loc[gdf_trans["End_poly"] == ind, "Start_poly"])
        ]
        gdf_voronoi.loc[ind, "Cap"] = gdf_trans.loc[(gdf_trans["Start_poly"] == ind) | (gdf_trans["End_poly"] == ind), "Cap_MVA"].sum()
    gdf_voronoi["Ratio"] = gdf_voronoi["Cap"] / gdf_voronoi["Area"]

    # Merge electric numbers with highest ratios in order to reduce the total number of polygons to the desired number of clusters
    islands = 0
    gdf_trans.set_index(["ID"], inplace=True)
    compression = len(gdf_voronoi.index) - param["number_clusters"]
    gdf_voronoi["Cluster"] = gdf_voronoi["ID_polygon"]

    print("Clustering regions based on connectivity")
    while len(set(gdf_voronoi["Cluster"].values)) > param["number_clusters"]:
        # Progress bar
        display_progress("", [compression, compression - len(set(gdf_voronoi["Cluster"].values)) + param["number_clusters"]])

        # Reset Ratio to zero for very large areas
        gdf_voronoi.loc[gdf_voronoi["Area"] > (total_area / param["number_clusters"]), "Ratio"] = 0

        # Get the index of the polygon with the highest ratio
        poly1_index = gdf_voronoi["Ratio"].idxmax()
        poly1 = gdf_voronoi.loc[poly1_index]

        # Get the dataframe of electric neighbors
        poly1_neighbors = gdf_voronoi[gdf_voronoi["Cluster"].isin(poly1["elec_neighbors"])]

        # Get the index of polygon from electric neighbors which has the highest ratio. It is named poly2
        poly2_index = poly1_neighbors["Ratio"].idxmax()
        poly2 = poly1_neighbors.loc[poly2_index]
        poly2_neighbors = gdf_voronoi[gdf_voronoi["Cluster"].isin(poly2["elec_neighbors"])]

        # Assign cluster number
        cluster_no = min(poly1["Cluster"], poly2["Cluster"])
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == poly1["Cluster"], "Cluster"] = cluster_no
        gdf_voronoi.loc[gdf_voronoi["Cluster"] == poly2["Cluster"], "Cluster"] = cluster_no

        gdf_trans, gdf_voronoi = update_values_in_geodataframes(gdf_trans, gdf_voronoi, poly1, poly2, cluster_no)

        # Check if there are no electric neighbors for the dissolved polygon (island)
        for isl in gdf_voronoi.loc[(gdf_voronoi["Cap"] == 0) & (gdf_voronoi["Cluster"] > 0)].index:
            islands = islands + 1
            gdf_voronoi.loc[isl, "Cluster"] = -islands

        # Dissolve geometries based on cluster number
        gdf_voronoi = gdf_voronoi.dissolve(by=["Cluster"]).reset_index()
        gdf_voronoi.index = gdf_voronoi["ID_polygon"]

        for ind in gdf_voronoi.loc[gdf_voronoi["Cluster"] < 0].index:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ws = ps.weights.Queen.from_dataframe(gdf_voronoi)
                ind_neighbor = gdf_voronoi.loc[ws[ind].keys(), "Area"].idxmax()
                poly1 = gdf_voronoi.loc[ind]
                poly2 = gdf_voronoi.loc[ind_neighbor]
                cluster_no = ind_neighbor
                gdf_voronoi.loc[ind_neighbor, "Area"] = gdf_voronoi.loc[[ind, ind_neighbor], "Area"].sum()
                gdf_voronoi.loc[ind_neighbor, "Ratio"] = gdf_voronoi.loc[ind_neighbor, "Cap"] / gdf_voronoi.loc[ind_neighbor, "Area"]
                gdf_voronoi.loc[ind, ["Cluster", "ID_polygon", "elec_neighbors", "trans_lines", "Cap"]] = gdf_voronoi.loc[
                    ind_neighbor, ["Cluster", "ID_polygon", "elec_neighbors", "trans_lines", "Cap"]
                ]
                gdf_voronoi = gpd.GeoDataFrame(gdf_voronoi, geometry="geometry", crs=param["CRS_grid"])
                gdf_voronoi = gdf_voronoi.dissolve(by=["Cluster"]).reset_index()
                gdf_voronoi.index = gdf_voronoi["ID_polygon"]
                gdf_trans, gdf_voronoi = update_values_in_geodataframes(gdf_trans, gdf_voronoi, poly1, poly2, cluster_no)
            except:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    knnw = ps.weights.KNN.from_dataframe(gdf_voronoi, k=1)
                    ws = libpysal.weights.util.attach_islands(ws, knnw)
                ind_neighbor = gdf_voronoi.loc[ws[ind].keys(), "Area"].idxmax()
                poly1 = gdf_voronoi.loc[ind]
                poly2 = gdf_voronoi.loc[ind_neighbor]
                cluster_no = ind_neighbor
                gdf_voronoi.loc[ind_neighbor, "Area"] = gdf_voronoi.loc[[ind, ind_neighbor], "Area"].sum()
                gdf_voronoi.loc[ind_neighbor, "Ratio"] = gdf_voronoi.loc[ind_neighbor, "Cap"] / gdf_voronoi.loc[ind_neighbor, "Area"]
                gdf_voronoi.loc[ind, ["Cluster", "ID_polygon", "elec_neighbors", "trans_lines", "Cap"]] = gdf_voronoi.loc[
                    ind_neighbor, ["Cluster", "ID_polygon", "elec_neighbors", "trans_lines", "Cap"]
                ]
                gdf_voronoi = gdf_voronoi.dissolve(by=["Cluster"]).reset_index()
                gdf_voronoi.index = gdf_voronoi["ID_polygon"]
                gdf_trans, gdf_voronoi = update_values_in_geodataframes(gdf_trans, gdf_voronoi, poly1, poly2, cluster_no)

        if len(set(gdf_voronoi["Cluster"].values)) in param["intermediate_number"]:
            paths["grid_intermediate"] = paths["lines_clustering"] + "grid_clusters_" + str(len(set(gdf_voronoi["Cluster"].values))) + ".shp"
            gdf_voronoi.drop(columns=["elec_neighbors", "trans_lines", "ID_polygon"]).to_file(
                driver="ESRI Shapefile", filename=paths["grid_intermediate"]
            )

    # Save results
    gdf_voronoi.drop(columns=["elec_neighbors", "trans_lines", "ID_polygon"]).to_file(driver="ESRI Shapefile", filename=paths["grid_regions"])

    print("\n")
    timecheck("End")

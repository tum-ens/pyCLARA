from .spatial_functions import assign_disconnected_components_to_nearest_neighbor
from .util import *


def max_p_clustering(paths, param):
    """
    This function applies the max-p algorithm to the obtained polygons. Depending on the number of clusters in the whole map after k-means,
    it decides whether to run max-p clustering multiple times (for each part, then for the whole map) or once (for the whole map).
    If you have already results for each part, you can skip that by setting *use_results_of_max_parts* to 1.
    
    :param paths: Dictionary of paths pointing to *polygonized_clusters* and *max_p_combined*.
    :type paths: dict
    :param param: Dictionary of parameters including max-p related parameters (*maximum_number* and *use_results_of_maxp_parts*), and eventually the *compression_ratio*
      for the first round of max-p clustering.
    :type param: dict
    
    :return: The called functions :mod:`max_p_parts` and :mod:`max_p_whole_map` generate outputs.
    :rtype: None
    """
    timecheck("Start")

    # Get number of polygons after k-means
    all_polygons = gpd.read_file(paths["polygonized_clusters"])

    if (len(all_polygons) > param["maxp"]["maximum_number"]) and (param["maxp"]["use_results_of_maxp_parts"] == 0):
        # Two rounds of max-p
        param["compression_ratio"] = 0.9 * (param["maxp"]["maximum_number"] / len(all_polygons))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_p_parts(paths, param)
            max_p_whole_map(paths, param, paths["max_p_combined"])
    elif param["maxp"]["use_results_of_maxp_parts"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_p_whole_map(paths, param, paths["max_p_combined"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_p_whole_map(paths, param, paths["polygonized_clusters"])

    timecheck("End")


def max_p_parts(paths, param):
    """
    This function applies the max-p algorithm on each part. It identifies the neighbors from the shapefile of polygons for that part. If there are disconnected
    parts, it assumes that they are neighbors with the closest polygon.
    
    The max-p algorithm aggregates polygons to a maximum number of regions that fulfil a certain condition. That condition is formulated as a minimum share
    of the sum of data values, ``thr``. The threshold is set differently for each part, so that the largest and most diverse part keeps a large number of polygons,
    and the smallest and least diverse is aggregated into one region. This is determined by the function :mod:`get_coefficients`.
    
    After assigning the clusters to the polygons, they are dissolved according to them, and the values of each property are aggregated according to the aggregation
    functions of the inputs, saved in *agg*.
    
    :param paths: Dictionary containing the paths to the folder of *inputs* and *polygons*, to the CSV *non_empty_rasters*, to the output folder *parts_max_p* and to the
      output file *max_p_combined*.
    :type paths: dict
    :param param: Dictionary of parameters containing the *raster_names* and their *weights* and aggregation methods *agg*, the *compression_ratio* of the polygonized
      kmeans clusters, and the *CRS* to be used for the shapefiles.
    :type param: dict
    
    :return: The results of the clustering are shapefiles for each part, saved in the folder *parts_max_p*, and a combination of these for the whole map *max_p_combined*.
    :rtype: None
    """
    timecheck("Start")

    # Read all necessary inputs from CSV files for this function.
    df = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))

    # Group by part number, and calculate the product of rel_size and rel_std
    group_by_part = df.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(["part"]).prod()

    # Start max-p
    for i in non_empty_rasters:
        print("Running max-p for part: " + str(i))
        data = gpd.read_file(paths["polygons"] + "result_%d.shp" % i)

        # Calculate the weighted sum in 'Value', that will be used for clustering
        data["Value"] = 0
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            scaling = data[raster_name].mean()
            data["Value"] = data["Value"] + param["weights"][counter_files] * data[raster_name] / scaling

        # Create weights object
        w = ps.weights.Queen.from_shapefile(paths["polygons"] + "result_%d.shp" % i)
        w = assign_disconnected_components_to_nearest_neighbor(data, w)

        # Get coefficients for threshold equation
        coef = get_coefficients(paths)

        # Calculate threshold depending on the size and standard deviation
        thr = (
            (coef["A"] * (exp(-coef["B"] * (group_by_part.loc[i, "rel_size"] + (coef["C"] * group_by_part.loc[i, "rel_std"])))))
            * data["Value"].sum()
            * param["compression_ratio"]
            / 2
        )
        if len(data) == 1:
            thr = data["Value"].sum() - 0.01

        rd.seed(100)
        np.random.seed(100)
        r = ps.region.maxp.Maxp(w, data["Value"].values.reshape(-1, 1), floor=thr, floor_variable=data["Value"], initial=100)
        print("Number of clusters after max-p: " + str(r.p) + " (before: " + str(len(data)) + ")")

        if r.p == 0:
            import pdb

            # Try to figure out the problem - might not occur anymore in newest version
            pdb.set_trace()

        data["CL"] = pd.Series(r.area2region).reindex(data.index)
        data["geometry"] = data["geometry"].buffer(0)

        # Calculating the area of each cluster using Lambert Cylindrical Equal Area EPSG:9835 (useful for the density, but needs a projection)
        if param["CRS"] == "epsg:4326":
            data.to_crs("+proj=cea")
            data["area"] = data["geometry"].area / 10 ** 6
            data.to_crs(epsg=4326)
        else:
            data["area"] = data["geometry"].area / 10 ** 6

        # Calculate the aggregated values for each cluster based on the aggregation method
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            if param["agg"][counter_files] == "density":
                data[raster_name] = data[raster_name] * data["area"]
                for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                    data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].sum() / data.loc[data["CL"] == cl, "area"].sum()
            if param["agg"][counter_files] == "mean":
                for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                    data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].mean()
            elif param["agg"][counter_files] == "sum":
                for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                    data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].sum()

        file = data.dissolve(by="CL")
        file.reset_index(inplace=True)

        # Result for every part after max-p one
        file.to_file(driver="ESRI Shapefile", filename=paths["parts_max_p"] + "max_p_part_%d.shp" % i)

    print("Merging all parts of max-p 1")
    gdf = gpd.read_file(paths["parts_max_p"] + "max_p_part_%d.shp" % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths["parts_max_p"] + "max_p_part_%d.shp" % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf["CL"] = gdf.index
    gdf["geometry"] = gdf.buffer(0)
    gdf.to_file(driver="ESRI Shapefile", filename=paths["max_p_combined"])

    timecheck("End")


def max_p_whole_map(paths, param, combined_file):
    """
    This function runs the max-p algorithm for the whole map, either on the results obtained from :mod:`max_p_parts`, or on those obtained from :mod:`polygonize_after_k_means`,
    depending on the number of polygons after kmeans clustering.
    
    It identifies the neighbors from the shapefile of polygons. If there are disconnected components (an island of polygons), it assumes that they are neighbors
    with the closest polygon. It also verifies that the code identifies neighbors properly and corrects that eventually using :mod:`correct_neighbors_in_shapefile`.
    
    The max-p algorithm aggregates polygons to a maximum number of regions that fulfil a certain condition. That condition is formulated as a minimum share
    of the sum of data values, ``thr``. The threshold for the whole map is set as a function of the number of polygons before clustering, and the desired number of polygons
    at the end. However, that number may not be matched exactly. The user may wish to adjust the threshold manually until the desired number is reached (increase the threshold
    to reduce the number of regions, and vice versa).
    
    After assigning the clusters to the polygons, they are dissolved according to them, and the values of each property are aggregated according to the aggregation
    functions of the inputs, saved in *agg*.
    
    :param paths: Dictionary containing the paths to the folder of *inputs* and to the output file *output*.
    :type paths: dict
    :param param: Dictionary of parameters containing the *raster_names* and their *weights* and aggregation methods *agg*, the desired number of features at the end
      *final_number*, and the *CRS* to be used for the shapefiles.
    :type param: dict
    :param combined_file: Path to the shapefile to use as input. It is either the result obtained from :mod:`max_p_parts`, or the one obtained from :mod:`polygonize_after_k_means`.
    :type combined_file: str
    
    :return: The result of the clustering is one shapefile for the whole map saved directly in *output*.
    :rtype: None
    """
    timecheck("Start")

    # Read combined file with regions to be clustered
    data = gpd.read_file(combined_file)

    # Calculate the weighted sum in 'Value', that will be used for clustering
    data["Value"] = 0
    for counter_files in range(len(paths["inputs"])):
        raster_name = param["raster_names"].split(" - ")[counter_files][:10]
        scaling = data[raster_name].mean()
        data["Value"] = data["Value"] + param["weights"][counter_files] * data[raster_name] / scaling

    # Create weights object
    w = ps.weights.Queen.from_shapefile(combined_file)
    w = assign_disconnected_components_to_nearest_neighbor(data, w)

    # Correcting neighbors
    print("Correcting neighbors.")
    w.neighbors = correct_neighbors_in_shapefile(param, combined_file, w.neighbors)
    print("Neighbors corrected!")

    thr = 0.0026 * (len(data) / param["maxp"]["final_number"]) * data["Value"].sum()
    # thr = 0.3 * (param["maxp"]["final_number"] / len(data)) * data["Value"].sum()
    print("Threshold: " + str(thr))
    random_no = rd.randint(1000, 1500)  # The range is selected randomly.
    np.random.seed(random_no)

    r = ps.region.maxp.Maxp(w, data["Value"].values.reshape(-1, 1), floor=thr, floor_variable=data["Value"], initial=100)
    print("Max-p finished!")
    print("Number of clusters: " + str(r.p))

    data["CL"] = pd.Series(r.area2region).reindex(data.index)
    data["geometry"] = data.buffer(0)

    # Calculating the area of each cluster using Lambert Cylindrical Equal Area EPSG:9835 (useful for the density, but needs a projection)
    if param["CRS"] == "epsg:4326":
        data.to_crs("+proj=cea")
        data["area"] = data["geometry"].area / 10 ** 6
        data.to_crs(epsg=4326)
    else:
        data["area"] = data["geometry"].area / 10 ** 6

    # Calculate the aggregated values for each cluster based on the aggregation method
    for counter_files in range(len(paths["inputs"])):
        raster_name = param["raster_names"].split(" - ")[counter_files][:10]
        if param["agg"][counter_files] == "density":
            data[raster_name] = data[raster_name] * data["area"]
            for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].sum() / data.loc[data["CL"] == cl, "area"].sum()
        if param["agg"][counter_files] == "mean":
            for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].mean()
        elif param["agg"][counter_files] == "sum":
            for cl in data.loc[pd.notnull(data[raster_name]), "CL"].unique():
                data.loc[data["CL"] == cl, raster_name] = data.loc[data["CL"] == cl, raster_name].sum()

    output = data.dissolve(by="CL")
    output.reset_index(inplace=True)

    output["NAME_SHORT"] = ["CL" + str(output.loc[i, "CL"]).zfill(2) for i in output.index]
    output.crs = {"init": param["CRS"]}
    output.to_file(driver="ESRI Shapefile", filename=paths["output"])  # Final file

    timecheck("End")


def correct_neighbors_in_shapefile(param, combined_file, existing_neighbors):
    """
    This function finds the neighbors in the shapefile. Somehow, max-p cannot figure out the correct neighbors and
    some clusters are physically neighbors but they are not considered as neighbors. This is where this function comes
    in. 
    
    It creates a small buffer around each polygon. If the enlarged polygons intersect, and the area of the intersection exceeds a threshold, then
    the polygons are considered neighbors, and the dictionary of neighbors is updated.

    :param param: The dictionary of parameters including the coordinate reference system *CRS* and the resolution of input rasters *res_desired*.
    :type param: dict
    :param combined_file: The path to the shapefile of polygons to be clustered.
    :type combined_file: str
    :param existing_neighbors: The dictionary of neighbors as extracted from the shapefile, before any eventual correction.
    :type existing_neighbors: dict
    
    :return neighbors_corrected: The dictionary of neighbors after correction (equivalent to an adjacency matrix).
    :rtype: dict
    """

    df = gpd.read_file(combined_file)

    # Create copy and project it using Lambert Cylindrical Equal Area EPSG:9835, if no projection given
    df_copy = df.copy()
    df_copy.crs = {"init": param["CRS"]}
    if param["CRS"] == "epsg:4326":
        df_copy.to_crs("+proj=cea")

    df["NEIGHBORS"] = None
    for index, cluster in df.iterrows():
        cluster_obj = gpd.GeoDataFrame(cluster.to_frame().T, geometry="geometry")
        cluster_obj.crs = {"init": param["CRS"]}

        cluster_obj["geometry"] = cluster_obj["geometry"].buffer(3 * param["res_desired"][0])  # in map units

        if param["CRS"] == "epsg:4326":
            cluster_obj.to_crs("+proj=cea")
        intersection = gpd.overlay(df_copy, cluster_obj, how="intersection")
        intersection["area"] = intersection["geometry"].area / 10 ** 6  # in km²
        intersection = intersection[intersection["area"] > 0.01]  # avoids that neighbors share only a point or a very small area
        neighbors = intersection.CL_1.tolist()
        # Remove own name from the list
        try:
            neighbors.remove(cluster_obj["CL"])
        except ValueError:
            pass
        # Add names of neighbors as NEIGHBORS value
        df.loc[index, "NEIGHBORS"] = ",".join(str(n) for n in neighbors)

    # Make the w.neighbors dictionary for replacing it in max_p_whole_map
    neighbors_corrected = dict()
    for index, cluster in df.iterrows():
        neighbors_obj = cluster["NEIGHBORS"].split(",")
        neighbors_int = list()
        for neighbor in neighbors_obj:
            if neighbor:
                neighbors_int.append(int(neighbor))
        neighbors_corrected[index] = neighbors_int
        for value in existing_neighbors[index]:
            if value not in neighbors_corrected[index]:
                neighbors_corrected[index].append(value)
        neighbors_corrected[index] = sorted(neighbors_corrected[index])

    return neighbors_corrected


def get_coefficients(paths):
    """
    This function gets the coefficients A, B and C for solving the 3 equations which will lead to the calculation
    of the threshold in the max-p algorithm.

    :param paths: The dictionary of paths including the one to *non_empty_rasters*.
    :type paths: str

    :return coef: The coefficient values for A, B and C returned as a dictionary. The expected structure is similar to this dictionary: {'A': 0.55, 'B': 2.91, 'C': 0.61}.
    :rtype: dict
    """
    # Get coefficients for threshold equation
    ul_point, ur_point, ll_point, lr_point = get_x_y_values(paths)
    est = [1, 1, 1]

    # Run equation solver
    coef = fsolve(eq_solver, est, args=(ll_point, ul_point, ur_point), xtol=0.001)

    coef_dict = {"A": coef[0], "B": coef[1], "C": coef[2]}
    return coef_dict


def eq_solver(coef, ll_point, ul_point, ur_point):
    """
    This function serves as the solver to find coefficient values A, B and C for our defined function which is used to
    calculate the threshold.

    :param coef: The coefficients which are calculated.
    :type coef: dict
    :param ll_point: Coordinates of lower left point.
    :type ll_point: tuple(int, int)
    :param ul_point: Coordinates of upper left point.
    :type ul_point: tuple(int, int)
    :param ur_point: Coordinates of upper right point.
    :type ur_point: tuple(int, int)

    :return f: Coefficient values for A, B and C in a numpy array. A is f[0], B is f[1] and C is f[2].
    :rtype: numpy array
    """
    A = coef[0]
    B = coef[1]
    C = coef[2]

    f = np.zeros(3)

    f[0] = (A * (exp(-B * (ll_point[0] + (C * ll_point[1]))))) - 0.5
    # f[1] = (A * (exp(-B * (ul_point[0] + (C * ul_point[1]))))) - 0.1
    f[2] = (A * (exp(-B * (ur_point[0] + (C * ur_point[1]))))) - 0.01

    return f

from lib.spatial_functions import *
from lib.util import *


def calculate_stats_for_non_empty_rasters(paths, param):
    """
    This function ...
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    timecheck("Start")

    non_empty_rasters = pd.DataFrame(
        columns=["no_columns", "no_rows", "size", "std", "rel_size", "rel_std", "prod_size_std"],
        index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=[u"file", u"part"]),
    )

    # Reading CSV file to get total parts of map
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)
    parts_of_map = int(df.loc["map_parts_total", "value"])

    print("Getting size and std of every raster part.")
    for counter_files in range(len(paths["inputs"])):
        input_file = paths["inputs"][counter_files]
        raster_name = param["raster_names"].split(" - ")[counter_files]

        for i in range(1, parts_of_map + 1):
            file = paths["sub_rasters"] + raster_name + "_sub_part_%d.tif" % i
            dataset = gdal.Open(file)
            band_raster = dataset.GetRasterBand(1)
            array_raster = band_raster.ReadAsArray()
            # Get number of columns and rows (needed later)
            non_empty_rasters.loc[(raster_name, i), ["no_columns", "no_rows"]] = (array_raster.shape[1], array_raster.shape[0])
            # Replace non-valid values with NaN
            array_raster[np.isnan(array_raster)] = param["minimum_valid"] - 1
            array_raster[array_raster < param["minimum_valid"]] = np.nan
            if np.sum(~np.isnan(array_raster)) == 0:
                continue

            array_raster = array_raster.flatten()
            array_raster = array_raster[~np.isnan(array_raster)]

            size_raster = len(array_raster)
            std_raster = array_raster.std(axis=0)

            non_empty_rasters.loc[(raster_name, i), ["size", "std"]] = (size_raster, std_raster)

        print(
            "Calculating relative size, relative std, product of relative size and relative std and four extreme corners of data cloud for raster "
            + raster_name
        )
        non_empty_rasters["rel_size"] = non_empty_rasters["size"] / non_empty_rasters["size"].max()
        non_empty_rasters["rel_std"] = non_empty_rasters["std"] / non_empty_rasters["std"].max()
        non_empty_rasters["prod_size_std"] = non_empty_rasters["rel_size"] * non_empty_rasters["rel_std"]
        non_empty_rasters["ul_corner"] = np.exp((-non_empty_rasters["rel_size"] + non_empty_rasters["rel_std"]).astype(float))
        non_empty_rasters["ur_corner"] = np.exp((non_empty_rasters["rel_size"] + non_empty_rasters["rel_std"]).astype(float))
        non_empty_rasters["ll_corner"] = np.exp((-non_empty_rasters["rel_size"] - non_empty_rasters["rel_std"]).astype(float))
        non_empty_rasters["lr_corner"] = np.exp((non_empty_rasters["rel_size"] - non_empty_rasters["rel_std"]).astype(float))

    # Write the numbers of non-empty raster files to CSV
    non_empty_rasters = non_empty_rasters.astype("float64")
    non_empty_rasters.to_csv(paths["non_empty_rasters"], sep=";", decimal=",")
    print("File saved: " + paths["non_empty_rasters"])

    # Write the data related to the maximum size and std to input_stats file for further use
    df.loc["size_max", "value"] = non_empty_rasters["size"].max()
    df.loc["std_max", "value"] = non_empty_rasters["std"].max()
    df.to_csv(paths["input_stats"], sep=";", decimal=",")
    print("\n")

    timecheck("End")


def choose_ref_part(paths, param):
    """
    This function chooses the reference part for the function identify_number_of_optimum_clusters.
    The reference part is chosen based on the product of relative size and relative standard deviation.
    The part with the largest product in all the input files is chosen.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """

    # Read CSV files
    non_empty_rasters = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)

    # Find the part with the maximum relative size x relative std.
    group_by_part = non_empty_rasters.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(["part"]).prod()
    ref_part = group_by_part.loc[group_by_part["prod_size_std"] == group_by_part["prod_size_std"].max()].index.values[0]
    print("The chosen reference part is: sub_part_" + str(ref_part) + ".tif")

    # Write the data related to the reference part to input_stats file for further use
    df.loc["ref_part_name", "value"] = ref_part
    df.to_csv(paths["input_stats"], sep=";", decimal=",")


def identify_opt_number_of_clusters(paths, param, part, size_of_raster, std_of_raster):
    """
    This function identifies number of optimum clusters which will be chosen for k-means
    Further explanation:
    Standard deviation and size of this reference part are used to estimate the no of clusters of every other part.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    # This function is used to determine the optimum number of clusters for respective part

    # Read all necessary inputs from input_stats
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)
    size_max = df.loc["size_max", "value"]
    std_max = df.loc["std_max", "value"]

    coef_std = 1 / (1 + param["kmeans"]["ratio_size_to_std"])
    coef_size = 1 - coef_std

    if param["kmeans"]["method"] == "reference_part":
        maximum_no_of_clusters = int(df.loc["max_no_of_cl_ref", "value"])
        optimum_no_of_clusters_for_raster = int(
            np.ceil(maximum_no_of_clusters * (coef_size * (size_of_raster / size_max) + coef_std * (std_of_raster / std_max)))
        )
    elif param["kmeans"]["method"] == "maximum_number":
        # Read the CSV file non_empty_rasters
        non_empty_rasters = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])

        # Group by part
        group_by_part = non_empty_rasters.reset_index(inplace=False)
        group_by_part = group_by_part[["part", "rel_size", "rel_std"]].groupby(["part"]).max()
        group_by_part["coefficient"] = coef_size * group_by_part["rel_size"] + coef_std * group_by_part["rel_std"]
        group_by_part["share"] = group_by_part["coefficient"] / group_by_part["coefficient"].sum()

        maximum_no_of_clusters = int(df.loc["max_no_of_cl_total", "value"])
        optimum_no_of_clusters_for_raster = int(np.ceil(group_by_part.loc[part, "share"] * maximum_no_of_clusters))

    if std_of_raster == 0:
        optimum_no_of_clusters_for_raster = 1
    if size_of_raster < optimum_no_of_clusters_for_raster:
        optimum_no_of_clusters_for_raster = size_of_raster
    print("Optimum number of clusters for part " + str(part) + " is " + str(optimum_no_of_clusters_for_raster))

    return optimum_no_of_clusters_for_raster


def identify_max_number_of_clusters_in_ref_part(paths, param):
    """
    This function identifies number of optimum clusters which will be chosen for k-means
    Further explanation:
    Standard deviation and size of this reference part are used to estimate the no of clusters of every other part.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    timecheck("Start")

    # Read input_stats.csv
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)

    ref_part_no = int(df.loc["ref_part_name", "value"])
    data = pd.DataFrame()
    for counter_files in range(len(paths["inputs"])):
        raster_name = param["raster_names"].split(" - ")[counter_files]
        reference_part = paths["sub_rasters"] + raster_name + "_sub_part_" + str(ref_part_no) + ".tif"
        # Open reference part as a dataset
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
        if len(data) == 0:
            data = pd.DataFrame({"X": X, "Y": Y, raster_name: array_raster}).set_index(["X", "Y"])
        else:
            data = data.join(pd.DataFrame({"X": X, "Y": Y, raster_name: array_raster}).set_index(["X", "Y"]), how="inner")

    coef = data.copy()
    coef.reset_index(inplace=True)
    coef["X"] = (coef["X"] - coef["X"].min()) / (coef["X"].max() - coef["X"].min())
    coef["Y"] = (coef["Y"] - coef["Y"].min()) / (coef["Y"].max() - coef["Y"].min())
    n_cols = len(paths["inputs"])
    # The values in the other columns are given less weight
    for counter_files in range(len(paths["inputs"])):
        raster_name = param["raster_names"].split(" - ")[counter_files]
        if coef[raster_name].min() == coef[raster_name].max():
            coef[raster_name] = 0.1 / np.sqrt(n_cols) * param["weights"][counter_files]
        else:
            coef[raster_name] = (
                0.1
                / np.sqrt(n_cols)
                * param["weights"][counter_files]
                * (coef[raster_name] - coef[raster_name].min())
                / (coef[raster_name].max() - coef[raster_name].min())
            )

    print("Running k-means in order to get the optimum number of clusters.")
    # This only needs to be run once
    k_means_stats = pd.DataFrame(columns=["Inertia", "Distance", "Slope"])
    k_min = param["kmeans"]["reference_part"]["min"]
    k_max = param["kmeans"]["reference_part"]["max"]
    k_step = param["kmeans"]["reference_part"]["step"]
    k_set = [k_min, k_max]
    k_set.extend(range(k_min + k_step, k_max - k_step + 1, k_step))
    for i in k_set:
        print("Checking for cluster number:" + str(i))
        kmeans = sklearn.cluster.KMeans(
            n_clusters=i,
            init="k-means++",
            n_init=10,
            max_iter=1000,
            tol=0.0001,
            precompute_distances="auto",
            verbose=0,
            copy_x=True,
            n_jobs=param["n_jobs"],
            algorithm="auto",
        )
        CL = kmeans.fit(coef)
        k_means_stats.loc[i, "Inertia"] = kmeans.inertia_  # inertia is the sum of the square of the euclidean distances
        print("Inertia: ", kmeans.inertia_)

        p = OptimumPoint((i - k_min) // k_step + 1, k_means_stats.loc[i, "Inertia"])
        if i == k_set[0]:
            p1 = p
            k_means_stats.loc[i, "Distance"] = 0
        elif i == k_set[1]:
            p2 = p
            k_means_stats.loc[i, "Distance"] = 0
        else:
            k_means_stats.loc[i, "Distance"] = p.distance_to_line(p1, p2)
            k_means_stats.loc[i, "Slope"] = k_means_stats.loc[i, "Distance"] - k_means_stats.loc[i - k_step, "Distance"]
            if abs(k_means_stats.loc[i, "Slope"]) <= 0.2:
                break

    k_means_stats = k_means_stats.astype("float64")
    k_means_stats.to_csv(paths["kmeans_stats"], index=False, sep=";", decimal=",")

    # The point for which the slope is less than threshold is taken as optimum number of clusters
    maximum_number_of_clusters_ref_part = int(i)
    print("Number of maximum clusters: " + str(maximum_number_of_clusters_ref_part))

    # Write the number of maximum clusters to csv file for further use
    df.loc["max_no_of_cl_ref", "value"] = maximum_number_of_clusters_ref_part
    df.to_csv(paths["input_stats"], sep=";", decimal=",")

    timecheck("End")


def k_means_clustering(paths, param):
    """
    This function does the k-means clustering for every part.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    timecheck("Start")

    # Read all necessary inputs from input_stats
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)

    if param["kmeans"]["method"] == "reference_part":
        df.loc["max_no_of_cl_total", "value"] = np.nan
    elif param["kmeans"]["method"] == "maximum_number":
        df.loc["max_no_of_cl_ref", "value"] = np.nan
        df.loc["max_no_of_cl_total", "value"] = param["kmeans"]["maximum_number"]
    else:
        print("Warning")

    # Update input_stats
    df.to_csv(paths["input_stats"], sep=";", decimal=",")

    parts_of_map = int(df.loc["map_parts_total", "value"])
    no_of_columns_in_map = int(df.loc["output_raster_columns", "value"])
    no_of_rows_in_map = int(df.loc["output_raster_rows", "value"])

    # Read the indices of non empty rasters from non_empty_rasters
    df = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))

    # Applying k-means on all parts
    for i in non_empty_rasters:
        print("Running k-means on part " + str(i))
        data = pd.DataFrame()
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files]

            # Open raster file as dataset for conversion to array for k-means
            file = paths["sub_rasters"] + raster_name + "_sub_part_%d.tif" % i
            dataset = gdal.Open(file)

            (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
            band_raster = dataset.GetRasterBand(1)
            array_raster = band_raster.ReadAsArray()
            array_raster[np.isnan(array_raster)] = param["minimum_valid"] - 1
            array_raster[array_raster < param["minimum_valid"]] = np.nan

            (y_index, x_index) = np.nonzero(~np.isnan(array_raster))
            X = x_index * x_size + upper_left_x + (x_size / 2)
            Y = y_index * y_size + upper_left_y + (y_size / 2)
            array_raster = array_raster.flatten()

            if len(data) == 0:
                table = pd.DataFrame({raster_name: array_raster})
                data = pd.DataFrame({"X": X, "Y": Y, raster_name: array_raster[~np.isnan(array_raster)]}).set_index(["X", "Y"])
                std_of_raster = data[raster_name].std(axis=0)
            else:
                table = table.join(pd.DataFrame({raster_name: array_raster}))
                data = data.join(
                    pd.DataFrame({"X": X, "Y": Y, raster_name: array_raster[~np.isnan(array_raster)]}).set_index(["X", "Y"]), how="inner"
                )
                std_of_raster = max(std_of_raster, data[raster_name].std(axis=0))

        coef = data.copy()
        coef.reset_index(inplace=True)
        coef["X"] = (coef["X"] - coef["X"].min()) / (coef["X"].max() - coef["X"].min())
        coef["Y"] = (coef["Y"] - coef["Y"].min()) / (coef["Y"].max() - coef["Y"].min())
        n_cols = len(paths["inputs"])
        # The values in the other columns are given less weight
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files]
            if coef[raster_name].min() == coef[raster_name].max():
                coef[raster_name] = 0.1 / np.sqrt(n_cols) * param["weights"][counter_files]
            else:
                coef[raster_name] = (
                    0.1
                    / np.sqrt(n_cols)
                    * param["weights"][counter_files]
                    * (coef[raster_name] - coef[raster_name].min())
                    / (coef[raster_name].max() - coef[raster_name].min())
                )

        size_of_raster = len(coef)

        # Determine the optimum number of clusters for respective part
        optimum_no_of_clusters_for_raster = identify_opt_number_of_clusters(paths, param, i, size_of_raster, std_of_raster)

        kmeans = sklearn.cluster.KMeans(
            n_clusters=optimum_no_of_clusters_for_raster,
            init="k-means++",
            n_init=10,
            max_iter=1000,
            tol=0.0001,
            precompute_distances="auto",
            verbose=0,
            copy_x=True,
            n_jobs=param["n_jobs"],
            algorithm="auto",
        )
        CL = kmeans.fit(coef)

        # Update x_index and y_index
        X = data.index.get_level_values(0)
        x_index = (X - upper_left_x - (x_size / 2)) / x_size
        x_index = list(np.round(x_index).astype(int))
        Y = data.index.get_level_values(1)
        y_index = (Y - upper_left_y - (y_size / 2)) / y_size
        y_index = list(np.round(y_index).astype(int))

        # Get number of columns and rows
        no_of_columns_in_map = int(df.loc[(raster_name, i), "no_columns"])
        no_of_rows_in_map = int(df.loc[(raster_name, i), "no_rows"])

        # Get cluster values
        clusters = np.empty([no_of_rows_in_map, no_of_columns_in_map])
        clusters[:] = param["minimum_valid"] - 1
        clusters[y_index, x_index] = CL.labels_
        # Convert array back to raster
        array_to_raster(clusters, paths["k_means"] + "clusters_part_%d.tif" % i, file)

        table["CL"] = clusters.flatten().astype(int)
        # Calculate the aggregated values for each cluster based on the aggregation method
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files]
            if (param["agg"][counter_files] == "density") or (param["agg"][counter_files] == "mean"):
                # For pixels with the same size, density = mean
                for cl in table.loc[pd.notnull(table[raster_name]), "CL"].unique():
                    table.loc[table["CL"] == cl, raster_name] = table.loc[table["CL"] == cl, raster_name].mean()
            elif param["agg"][counter_files] == "sum":
                for cl in table.loc[pd.notnull(table[raster_name]), "CL"].unique():
                    table.loc[table["CL"] == cl, raster_name] = table.loc[table["CL"] == cl, raster_name].sum()
            # Fill the rest with a value equivalent to NaN in this code
            table.loc[pd.isnull(table[raster_name]), raster_name] = param["minimum_valid"] - 1

        # Group by cluster number, then save the table for later
        table = table.groupby(["CL"]).mean()
        table.to_csv(paths["k_means"] + "clusters_part_%d.csv" % i, index=True, sep=";", decimal=",")
        print("k-means completed for raster part " + str(i) + "\n")

    timecheck("End")


class OptimumPoint:
    """
    This class is used in the elbow method to identify the maximum distance between the end point and the start point of
    the curve created between no. of clusters and inertia.
    """

    def __init__(self, init_x, init_y):
        self.x = init_x
        self.y = init_y

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
        den = sqrt(y_diff ** 2 + x_diff ** 2)
        return num / den


def polygonize_after_k_means(paths, param):
    """
    This function changes from raster after k-means to polygon layers which are used in max-p algorithm.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """

    # Read the indices of non empty rasters from non_empty_rasters
    df = pd.read_csv(paths["non_empty_rasters"], sep=";", decimal=",", index_col=[0, 1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))

    for i in non_empty_rasters:
        print("Polygonizing raster part: " + str(i))
        file_cluster = paths["k_means"] + "clusters_part_%d.tif" % i
        shape_cluster = paths["polygons"] + "clusters_part_%d.shp" % i
        polygonize_raster(file_cluster, shape_cluster, "CL")

        # Read table
        table = pd.read_csv(paths["k_means"] + "clusters_part_%d.csv" % i, index_col=0, sep=";", decimal=",")
        # Read shapefile with polygons
        file_cluster = gpd.read_file(shape_cluster)
        # Join shapefile and table
        file_cluster.set_index("CL", inplace=True)
        file_cluster = file_cluster.join(table)
        # Remove polygons of non valid points
        if param["minimum_valid"] - 1 in file_cluster.index:
            file_cluster.drop(param["minimum_valid"] - 1, axis=0, inplace=True)
        # Dissolve polygons within the same cluster
        file_cluster["geometry"] = file_cluster["geometry"].buffer(0)
        file_cluster = file_cluster.dissolve(by="CL")
        # Reset index and CRS
        file_cluster.reset_index(inplace=True)
        file_cluster.crs = {"init": param["CRS"]}
        file_cluster.to_file(driver="ESRI Shapefile", filename=paths["polygons"] + "result_%d.shp" % i)

    # Merge all parts together after kmeans to see the output
    gdf = gpd.read_file(paths["polygons"] + "result_%d.shp" % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths["polygons"] + "result_%d.shp" % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf["CL"] = gdf.index
    gdf.crs = {"init": param["CRS"]}
    gdf.to_file(driver="ESRI Shapefile", filename=paths["polygonized_clusters"])

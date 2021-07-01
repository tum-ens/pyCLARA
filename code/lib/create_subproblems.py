from .spatial_functions import calc_region, array2raster, crd_bounding_box, ind_from_crd
from .util import *


def cut_raster(paths, param):
    """
    This function decides, based on the user preference in *use_shapefile*, whether to call the
    module :mod:`cut_raster_using_shapefile` or :mod:`cut_raster_using_boxes`.
    
    :param paths: Dictionary of paths to inputs and outputs.
    :type paths: dict
    :param param: Dictionary of parameters and user preferences.
    :type param: dict
    
    :return: The submodules :mod:`cut_raster_using_shapefile` and :mod:`cut_raster_using_boxes` have their own outputs.
    :rtype: None
    """
    if param["use_shapefile"]:
        cut_raster_using_shapefile(paths, param)
    else:
        cut_raster_using_boxes(paths, param)


def cut_raster_using_boxes(paths, param):
    """
    This function cuts the raster file into a m*n boxes with m rows and n columns.
    
    :param paths: The paths to the input rasters *inputs*, to the output raster parts *sub_rasters*, 
      and to the CSV file *input_stats* which will be updated.
    :type paths: dict
    :param param: Parameters that include the number of *rows* and *cols*, and the *raster_names*.
    :type param: dict
    
    :return: The raster parts are saved as GEOTIFF files, and the CSV file *input_stats* is updated.
    :rtype: None
    """

    timecheck("Start")

    scale_rows = param["rows"]
    scale_cols = param["cols"]

    for counter_files in range(len(paths["inputs"])):
        input_file = paths["inputs"][counter_files]
        raster_name = param["raster_names"].split(" - ")[counter_files]
        # Opening the raster file as a dataset
        dataset = gdal.Open(input_file)
        print("Dealing with raster file " + raster_name + " located in " + input_file)
        # The number of columns in raster file
        columns_in_raster_file = dataset.RasterXSize
        # The number of rows in raster file.
        rows_in_raster_file = dataset.RasterYSize

        # no of parts the map will be cut into.
        total_map_parts = scale_rows * scale_cols
        print("Total parts of map = " + str(total_map_parts))
        columns_in_output_raster = int(columns_in_raster_file / scale_cols)
        rows_in_output_raster = int(rows_in_raster_file / scale_rows)

        counter = 1
        gt = dataset.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]

        print("Cutting the raster " + raster_name + " into smaller boxes.")
        for i in range(1, scale_cols + 1):
            for j in range(1, scale_rows + 1):
                # cuts the input rasters into n equal parts according to the values assigned as parts_of_map,
                # columns_in_output_raster and rows_in_output_raster. gdal.Translate arguments are:(output_subset_file,
                # input_file, the 4 corners of the square which is to be cut).
                dc = gdal.Translate(
                    paths["sub_rasters"] + raster_name + "_sub_part_%d.tif" % counter,
                    dataset,
                    projWin=[
                        minx + (i - 1) * columns_in_output_raster * gt[1],
                        maxy - (j - 1) * rows_in_output_raster * gt[1],
                        minx + columns_in_output_raster * i * gt[1],
                        maxy - (j * rows_in_output_raster) * gt[1],
                    ],
                )

                print("Created part: " + raster_name + "_sub_part_" + str(counter))
                counter = counter + 1
        print("\n")

    # Writing the data related to map parts to input_stats.csv file for further use.
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)
    df.loc["map_parts_total", "value"] = total_map_parts
    df.loc["output_raster_columns", "value"] = columns_in_output_raster
    df.loc["output_raster_rows", "value"] = rows_in_output_raster
    df.to_csv(paths["input_stats"], sep=";", decimal=",")
    print("File updated: " + paths["input_stats"])

    timecheck("End")


def cut_raster_using_shapefile(paths, param):
    """
    This function cuts the raster file into parts using the features of a shapefile of (multi)polygons.
    Each feature is used as a mask, so that only the pixels lying on it are extracted and saved in a separate raster file.
    
    :param paths: The paths to the input rasters *inputs*, to the shapefile of *subregions*, to the output raster parts *sub_rasters*, 
      and to the CSV file *input_stats* which will be updated.
    :type paths: dict
    :param param: Parameters that include the array *Crd_all*, the array *res_desired* and the dictionary 
      *GeoRef* for georeferencing the output rasters, the minimum valid value *minimum_valid*, and the *raster_names*.
    :type param: dict
    
    :return: The raster parts are saved as GEOTIFF files, and the CSV file *input_stats* is updated.
    :rtype: None
    """

    timecheck("Start")

    Crd_all = param["Crd_all"]
    res_desired = param["res_desired"]
    GeoRef = param["GeoRef"]

    # Read shapefile of subregions
    subregions = gpd.read_file(paths["subregions"])
    subregions.reset_index(inplace=True, drop=True)

    for reg in range(len(subregions)):

        # Compute region_mask
        r = subregions.bounds.loc[reg]
        box = np.array([r["maxy"], r["maxx"], r["miny"], r["minx"]])[np.newaxis]
        Crd_reg = crd_bounding_box(box, res_desired)
        Ind_reg = ind_from_crd(Crd_reg, Crd_all, res_desired)
        A_subregion_extended = calc_region(subregions.loc[reg], Crd_all, res_desired, GeoRef)

        for counter_files in range(len(paths["inputs"])):
            input_file = paths["inputs"][counter_files]
            raster_name = param["raster_names"].split(" - ")[counter_files]

            # Opening the raster file as a dataset
            with rasterio.open(input_file) as src:
                dataset = np.flipud(src.read(1))

            # Calculate masked array
            dataset_masked = dataset * A_subregion_extended
            dataset_masked[A_subregion_extended == 0] = param["minimum_valid"] - 1
            dataset_masked = dataset_masked[Ind_reg[0, 2] : Ind_reg[0, 0], Ind_reg[0, 3] : Ind_reg[0, 1]]

            # Write masked array
            output_path = paths["sub_rasters"] + raster_name + "_sub_part_%d.tif" % (reg + 1)
            array2raster(output_path, dataset_masked, [Crd_reg[0, 3], Crd_reg[0, 0]], param)
            print("Created part: " + raster_name + "_sub_part_" + str(reg + 1))

    # Writing the data related to map parts to input_stats.csv file for further use.
    df = pd.read_csv(paths["input_stats"], sep=";", decimal=",", index_col=0)
    df.loc["map_parts_total", "value"] = len(subregions)
    df.loc["output_raster_columns", "value"] = A_subregion_extended.shape[1]
    df.loc["output_raster_rows", "value"] = A_subregion_extended.shape[0]
    df.to_csv(paths["input_stats"], sep=";", decimal=",")
    print("File updated: " + paths["input_stats"])

    timecheck("End")

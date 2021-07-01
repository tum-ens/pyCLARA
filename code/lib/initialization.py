from config import configuration
from .spatial_functions import calc_geotiff
from .util import *


def initialization():
    """
    This function reads the user-defined parameters and paths from :mod:`config.py`, then checks the validity of the input files. If they are missing or are not rasters,
    a warning is thrown and the code is exited. The same applies if the rasters do not have the same resolution or scope.
    
    If the input files are valid, the CSV file in *input_stats* is generated and filled with the information that is already available. It will be called by other functions
    and eventually filled by them.

    :return: The updated dictionaries param and paths.
    :rtype: tuple(dict, dict)
    """
    timecheck("Start")
    # import param and paths
    paths, param = configuration()

    # Check whether the inputs are correct
    if not len(paths["inputs"]):
        warn("No input file given", UserWarning)
        sys.exit(0)
    for input_file in paths["inputs"]:
        if not os.path.isfile(input_file):
            warn("File does not exist: " + input_file, UserWarning)
            sys.exit(0)
        elif not input_file.endswith(".tif"):
            warn("File is not raster: " + input_file, UserWarning)
            sys.exit(0)

    # Check that all rasters have the same scope and resolution
    for input_file in paths["inputs"]:
        dataset = gdal.Open(input_file)
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = dataset.GetGeoTransform()
        Crd_all = np.array(
            [[upper_left_y], [upper_left_x + x_size * dataset.RasterXSize], [upper_left_y + y_size * dataset.RasterYSize], [upper_left_x]]
        )
        if input_file == paths["inputs"][0]:
            Crd_all_old = Crd_all
            x_size_old = x_size
            y_size_old = y_size
        elif (Crd_all_old != Crd_all).any() or (x_size_old != x_size) or (y_size_old != y_size):
            warn("Not the same scope / resolution!", UserWarning)
            sys.exit(0)
        param["Crd_all"] = Crd_all
        param["res_desired"] = np.array([abs(x_size), abs(y_size)])
        param["GeoRef"] = calc_geotiff(Crd_all, param["res_desired"])

    # Create dataframe for input stats
    df = pd.DataFrame(
        index=[
            "map_parts_total",
            "output_raster_columns",
            "output_raster_rows",
            "ref_part_name",
            "size_max",
            "std_max",
            "max_no_of_cl_ref",
            "max_no_of_cl_total",
        ],
        columns=["value"],
    )
    df.loc[["output_raster_columns", "output_raster_rows"], "value"] = (dataset.RasterXSize, dataset.RasterYSize)
    if not os.path.exists(paths["input_stats"]):
        df.to_csv(paths["input_stats"], sep=";", decimal=",")
    timecheck("End")

    return paths, param

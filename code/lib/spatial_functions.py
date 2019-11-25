from lib.util import *

def array_to_raster(array, destination_file, input_raster_file):
    """
    This function changes from array back to raster (used after kmeans algorithm).
    
    :param array: The array which needs to be converted into a raster.
    :param destination_file: The file name with which the created raster file is saved.
    :param input_raster_file: The original input raster file from which the original coordinates are taken to convert the array back to raster.
    """
    source_raster = gdal.Open(input_raster_file)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = source_raster.GetGeoTransform()

    x_pixels = source_raster.RasterXSize  # number of pixels in x
    y_pixels = source_raster.RasterYSize  # number of pixels in y

    x_min = upper_left_x
    y_max = upper_left_y  # x_min & y_max are like the "top left" corner.
    wkt_projection = source_raster.GetProjection()
    driver = gdal.GetDriverByName('GTiff')

    # Write to disk
    dataset = driver.Create(destination_file, x_pixels, y_pixels, 1, gdal.GDT_Float32, ["COMPRESS=PACKBITS"])
    dataset.SetGeoTransform((x_min, x_size, 0, y_max, 0, y_size))
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()


def polygonize_raster(input_file, output_shapefile, column_name):
    """
    This function is used to change from a raster to polygons as max-p algorithm only works with polygons.
    
    :param input_file: The file which needs to be converted to a polygon from a raster.
    :param output_shapefile: The shape file which is generated after polygonization.
    :param column_name: The column name, the values from which are used for conversion.
    """
    
    source_raster = gdal.Open(input_file)
    band = source_raster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shapefile):
        driver.DeleteDataSource(output_shapefile)

    out_data_source = driver.CreateDataSource(output_shapefile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_raster.GetProjectionRef())
    out_layer = out_data_source.CreateLayer(output_shapefile, srs)
    new_field = ogr.FieldDefn(column_name, ogr.OFTInteger)
    out_layer.CreateField(new_field)
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    out_data_source.Destroy()


def create_voronoi_polygons(points_list):
    """
    This function makes voronoi polygons by taking a points list as input.
    :param points_list: The points list is used to make voronoi polygons.
    :return:
    """

    voronoi_polygons = Voronoi(points_list)

    # Make lines from voronoi polygons
    lines = [LineString(voronoi_polygons.vertices[line])
             for line in voronoi_polygons.ridge_vertices
             ]

    # Return list of polygons created from lines
    polygons_list = list()
    for polygon in polygonize(lines):
        polygons_list.append(polygon)

    return polygons_list
    
    
def calc_geotiff(Crd_all, res_desired):
    """
    This function returns a dictionary containing the georeferencing parameters for geotiff creation,
    based on the desired extent and resolution.

    :param Crd_all: Coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res_desired: Desired data resolution in the vertical and horizontal dimensions.
    :type res_desired: list

    :return GeoRef: Georeference dictionary containing *RasterOrigin*, *RasterOrigin_alt*, *pixelWidth*, and *pixelHeight*.
    :rtype: dict
    """
    GeoRef = {
        "RasterOrigin": [Crd_all[3], Crd_all[0]],
        "RasterOrigin_alt": [Crd_all[3], Crd_all[2]],
        "pixelWidth": res_desired[1],
        "pixelHeight": -res_desired[0],
    }
    return GeoRef
    

def crd_merra(Crd_regions, res_weather):
    """
    This function calculates coordinates of the bounding box covering MERRA-2 data.

    :param Crd_regions: Coordinates of the bounding boxes of the regions.
    :type Crd_regions: numpy array
    :param res_weather: Weather data resolution.
    :type res_weather: list

    :return Crd: Coordinates of the bounding box covering MERRA-2 data for each region.
    :rtype: numpy array
    """
    Crd = np.array(
        [
            np.ceil((Crd_regions[:, 0] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
            np.ceil(Crd_regions[:, 1] / res_weather[1]) * res_weather[1],
            np.floor((Crd_regions[:, 2] + res_weather[0] / 2) / res_weather[0]) * res_weather[0] - res_weather[0] / 2,
            np.floor(Crd_regions[:, 3] / res_weather[1]) * res_weather[1],
        ]
    )
    Crd = Crd.T
    return Crd
    

def ind_merra(Crd, Crd_all, res):
    """
    This function converts longitude and latitude coordinates into indices within the spatial scope of MERRA-2 data.

    :param Crd: Coordinates to be converted into indices.
    :type Crd: numpy array
    :param Crd_all: Coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res: Resolution of the data, for which the indices are produced.
    :type res: list
    
    :return Ind: Indices within the spatial scope of MERRA-2 data.
    :rtype: numpy array
    """
    if len(Crd.shape) == 1:
        Crd = Crd[np.newaxis]
    Ind = np.array(
        [
            (Crd[:, 0] - Crd_all[2]) / res[0],
            (Crd[:, 1] - Crd_all[3]) / res[1],
            (Crd[:, 2] - Crd_all[2]) / res[0] + 1,
            (Crd[:, 3] - Crd_all[3]) / res[1] + 1,
        ]
    )
    Ind = np.transpose(Ind).astype(int)
    return Ind
    

def calc_region(region, Crd_reg, res_desired, GeoRef):
    """
    This function reads the region geometry, and returns a masking raster equal to 1 for pixels within and 0 outside of
    the region.

    :param region: Region geometry
    :type region: Geopandas series
    :param Crd_reg: Coordinates of the region
    :type Crd_reg: list
    :param res_desired: Desired high resolution of the output raster
    :type res_desired: list
    :param GeoRef: Georeference dictionary containing *RasterOrigin*, *RasterOrigin_alt*, *pixelWidth*, and *pixelHeight*.
    :type GeoRef: dict

    :return A_region: Masking raster of the region.
    :rtype: numpy array
    """
    latlim = Crd_reg[2] - Crd_reg[0]
    lonlim = Crd_reg[3] - Crd_reg[1]
    M = int(math.fabs(latlim) / res_desired[0])
    N = int(math.fabs(lonlim) / res_desired[1])
    A_region = np.ones((M, N))
    origin = [Crd_reg[3], Crd_reg[2]]

    if region.geometry.geom_type == "MultiPolygon":
        features = [feature for feature in region.geometry]
    else:
        features = [region.geometry]
    west = origin[0]
    south = origin[1]
    profile = {
        "driver": "GTiff",
        "height": M,
        "width": N,
        "count": 1,
        "dtype": rasterio.float64,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(west, south, GeoRef["pixelWidth"], GeoRef["pixelHeight"]),
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as f:
            f.write(A_region, 1)
            out_image, out_transform = mask.mask(f, features, crop=False, nodata=0, all_touched=False, filled=True)
        A_region = out_image[0]

    return A_region
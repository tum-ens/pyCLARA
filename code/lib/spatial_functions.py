from .util import *


def array_to_raster(array, destination_file, input_raster_file):
    """
    This function changes from array back to raster (used after kmeans algorithm).
    
    :param array: The array which needs to be converted into a raster.
    :type array: numpy array
    :param destination_file: The file name where the created raster file is saved.
    :type destination_file: string
    :param input_raster_file: The original input raster file from which the original coordinates are taken to convert the array back to raster.
    :type input_raster_file: string
    
    :return: The raster file will be saved in the desired path *destination_file*.
    :rtype: None
    """
    source_raster = gdal.Open(input_raster_file)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = source_raster.GetGeoTransform()

    x_pixels = source_raster.RasterXSize  # number of pixels in x
    y_pixels = source_raster.RasterYSize  # number of pixels in y

    x_min = upper_left_x
    y_max = upper_left_y  # x_min & y_max are like the "top left" corner.
    wkt_projection = source_raster.GetProjection()
    driver = gdal.GetDriverByName("GTiff")

    # Write to disk
    dataset = driver.Create(destination_file, x_pixels, y_pixels, 1, gdal.GDT_Float32, ["COMPRESS=PACKBITS"])
    dataset.SetGeoTransform((x_min, x_size, 0, y_max, 0, y_size))
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()


def array2raster(newRasterfn, array, rasterOrigin, param):
    """
    This function saves array to geotiff raster format (used in cutting with shapefiles).

    :param newRasterfn: Output path of the raster.
    :type newRasterfn: string
    :param array: Array to be converted into a raster.
    :type array: numpy array
    :param rasterOrigin: Latitude and longitude of the Northwestern corner of the raster.
    :type rasterOrigin: list of two floats
    :param param: Dictionary of parameters including *GeoRef* and *CRS*.
    :type param: dict

    :return: The raster file will be saved in the desired path *newRasterfn*.
    :rtype: None
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64, ["COMPRESS=PACKBITS"])
    outRaster.SetGeoTransform((originX, param["GeoRef"]["pixelWidth"], 0, originY, 0, param["GeoRef"]["pixelHeight"]))
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(int(param["CRS"][5:]))
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(np.flipud(array))
    outband.FlushCache()
    outband = None


def polygonize_raster(input_file, output_shapefile, column_name):
    """
    This function is used to change from a raster to polygons as max-p algorithm only works with polygons.
    
    :param input_file: The path to the file which needs to be converted to a polygon from a raster.
    :type input_file: string
    :param output_shapefile: The path to the shapefile which is generated after polygonization.
    :type output_shapefile: string
    :param column_name: The column name, the values from which are used for conversion.
    :type column_name: string
    
    :return: The shapefile of (multi)polygons is saved directly in the desired path *output_shapefile*.
    :rtype: None
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


def crd_bounding_box(Crd_regions, resolution):
    """
    This function calculates coordinates of the bounding box covering data in a given resolution.

    :param Crd_regions: Coordinates of the bounding boxes of the regions.
    :type Crd_regions: numpy array
    :param resolution: Data resolution.
    :type resolution: numpy array

    :return Crd: Coordinates of the bounding box covering the data for each region.
    :rtype: numpy array
    """
    Crd = np.array(
        [
            np.ceil((Crd_regions[:, 0] + resolution[0] / 2) / resolution[0]) * resolution[0] - resolution[0] / 2,
            np.ceil(Crd_regions[:, 1] / resolution[1]) * resolution[1],
            np.floor((Crd_regions[:, 2] + resolution[0] / 2) / resolution[0]) * resolution[0] - resolution[0] / 2,
            np.floor(Crd_regions[:, 3] / resolution[1]) * resolution[1],
        ]
    )
    Crd = Crd.T
    return Crd


def ind_from_crd(Crd, Crd_all, res):
    """
    This function converts longitude and latitude coordinates into indices within the spatial scope of the data.

    :param Crd: Coordinates to be converted into indices.
    :type Crd: numpy array
    :param Crd_all: Coordinates of the bounding box of the spatial scope.
    :type Crd_all: numpy array
    :param res: Resolution of the data, for which the indices are produced.
    :type res: list
    
    :return Ind: Indices within the spatial scope of data.
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


def ckd_nearest(gdf_a, gdf_b, bcol):
    """
    This function finds the distance and the nearest points in gdf_b for every point in gdf_a.

    :param gdf_a: GeoDataFrame of points, forming a component that is disconnected from *gdf_b*.
    :type gdf_a: geodataframe
    :param gdf_b: GeoDataFrame of points, forming a component that is disconnected from *gdf_a*.
    :type gdf_b: geodataframe
    :param bcol: Name of column that should be listed in the resulting DataFrame.
    :type bcol: string

    :return df: Dataframe with the combinations of pair of points as rows, and ``'distance'`` and ``'bcol'`` as columns.
    :rtype: pandas dataframe
    """
    na = np.array(list(zip(gdf_a.geometry.x, gdf_a.geometry.y)))
    nb = np.array(list(zip(gdf_b.geometry.x, gdf_b.geometry.y)))
    btree = cKDTree(nb)
    dist, idx = btree.query(na, k=1)
    df = pd.DataFrame.from_dict({"distance": dist.astype(float), "bcol": gdf_b.loc[idx, bcol].values})

    return df


def assign_disconnected_components_to_nearest_neighbor(data, w):
    """
    This loop is used to force any disconnected group of polygons (graph component) to be assigned to the nearest neighbors.
    
    :param data: The geodataframe of polygons to be clustered.
    :type data: geodataframe
    :param w: The pysal weights object of the graph (``w.neighbors`` is similar to an adjacency matrix).
    :type w: pysal weights object
    
    :return w: The updated pysal weights objected is returned.
    :rtype: pysal weights object
    """

    if len(data) > 1:
        [n_components, labels] = cg.connected_components(w.sparse)
        if n_components > 1:
            # Attach islands if any to nearest neighbor
            knnw = ps.weights.KNN.from_dataframe(data, k=1)
            w = libpysal.weights.util.attach_islands(w, knnw)
            [n_components, labels] = cg.connected_components(w.sparse)

        if n_components > 1:
            # Disconnected areas exist. Removing them before max-p can be applied
            for comp in range(n_components):
                # Filter polygons within that component
                data_comp = data.loc[labels == comp]
                ind_comp = list(data_comp.index)

                data_comp.loc[ind_comp, "geometry"] = data_comp.loc[ind_comp, "geometry"].buffer(0)
                data_comp.loc[ind_comp, "dissolve_field"] = 1
                data_comp = data_comp.dissolve(by="dissolve_field")
                data_comp.index = [len(data)]

                data_new = data.drop(ind_comp)
                data_new = data_new.append(data_comp, sort=True)

                knnw = ps.weights.KNN.from_dataframe(data_new, k=1)
                for radius in range(1, len(data) - 1):
                    stop_condition = False
                    knn = ps.weights.KNN.from_dataframe(data, k=radius)
                    for ind in ind_comp:
                        if knn.neighbors[ind][radius - 1] == knnw.neighbors[len(data)][0]:
                            w.neighbors[ind] = w.neighbors[ind] + knnw.neighbors[len(data)]
                            w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[knnw.neighbors[len(data)][0]] + [ind]
                            stop_condition = True
                            continue
                    if stop_condition:
                        break

    return w

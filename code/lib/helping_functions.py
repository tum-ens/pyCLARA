


def array_to_raster(array, destination_file, input_raster_file):
    """This function changes from array back to raster (used after kmeans algorithm).
    :param array = The array which needs to be converted into a raster.
    :param destination_file = The file name with which the created raster file is saved.
    :param input_raster_file = The original input raster file from which the original coordinates are taken to convert
                               the array back to raster.
    """
    source_raster = gdal.Open(input_raster_file)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = source_raster.GetGeoTransform()

    x_pixels = source_raster.RasterXSize  # number of pixels in x
    y_pixels = source_raster.RasterYSize  # number of pixels in y

    x_min = upper_left_x
    y_max = upper_left_y  # x_min & y_max are like the "top left" corner.
    wkt_projection = source_raster.GetProjection()
    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(destination_file, x_pixels, y_pixels, 1, gdal.GDT_Float32, )
    dataset.SetGeoTransform((x_min, x_size, 0, y_max, 0, y_size))
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.

    #return dataset, dataset.GetRasterBand(1)  # If you need to return, remember to return


def polygonize(input_file, output_shape_file, column_name):
    """This function is used to change from a raster to polygons as max-p algorithm only works with polygons.
    :param input_file = The file which needs to be converted to a polygon from a raster.
    :param output_shape_file = The shape file which is generated after polygonization.
    :param column_name = The column name, the values from which are used for conversion.
    """
    
    source_raster = gdal.Open(input_file)
    band = source_raster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shape_file):
        driver.DeleteDataSource(output_shape_file)

    out_data_source = driver.CreateDataSource(output_shape_file)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_raster.GetProjectionRef())
    out_layer = out_data_source.CreateLayer(output_shape_file, srs)
    new_field = ogr.FieldDefn(column_name, ogr.OFTInteger)
    out_layer.CreateField(new_field)
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    out_data_source.Destroy()
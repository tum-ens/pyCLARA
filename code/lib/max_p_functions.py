from lib.util import *

def max_p_algorithm_new(paths, param):
    """
    This function applies the max-p algorithm to the obtained polygons.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    timecheck("Start")

    # Read all necessary inputs from CSV files for this function
    df = pd.read_csv(paths["non_empty_rasters"], sep=';', decimal=',', index_col=[0,1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))
    
    # Group by part number, and calculate the product of rel_size and rel_std
    group_by_part = df.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(['part']).prod()

    # Start max-p
    for i in non_empty_rasters:
        print('Running max-p for part: ' + paths["polygons"] + 'result_%d.shp' % i)
        data = gpd.read_file(paths["polygons"] + 'result_%d.shp' % i)
        
        # Calculate the weighted sum in 'Value', that will be used for clustering
        data['Value'] = 0
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            scaling = data[raster_name].mean()
            data['Value'] = data['Value'] + param["weights"][counter_files] * data[raster_name] / scaling

        # Create weights object
        w = ps.weights.Queen.from_shapefile(paths["polygons"] + 'result_%d.shp' % i)

        # This loop is used to force any disconnected group of polygons to be assigned to the nearest neighbors
        if len(data) > 1:
            knnw = ps.weights.KNN.from_shapefile(paths["polygons"] + 'result_%d.shp' % i, k=1)
            # Attach islands if any to nearest neighbor
            w = libpysal.weights.util.attach_islands(w, knnw)

            [n_components, labels] = cg.connected_components(w.sparse)
            if n_components > 1:
                print('Disconnected areas exist. Removing them before max-p can be applied.')
                for comp in range(n_components):
                    # Filter polygons within that component
                    data_comp = data.loc[labels == comp]
                    
                    import pdb; pdb.set_trace()
                    ss = [uu for uu, x in enumerate(labels == comp) if x]
                    dd = data.loc[ss]
                    dd['F'] = 1
                    dd['geometry'] = dd['geometry'].buffer(0)
                    dd = dd.dissolve(by='F')
                    dd.index = [len(data)]
                    dissolve = data.drop(ss)
                    dissolve = dissolve.append(dd)
                    knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                    for cc in range(1, len(data) - 1):
                        countern = 0
                        knn = ps.weights.KNN.from_dataframe(data, k=cc)
                        for s in range(len(ss)):
                            if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                                w.neighbors[ss[s]] = w.neighbors[ss[s]] + knnw.neighbors[len(data)]
                                w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[
                                                                                knnw.neighbors[len(data)][0]] + [
                                                                                ss[s]]
                                countern = countern + 1
                                continue
                        if countern > 0:
                            break
        # Get coefficients for threshold equation
        coef = get_coefficients(paths)

        # Calculate threshold depending on the size and standard deviation
        thr = (coef['A'] * (exp(-coef['B'] * (group_by_part.loc[i, 'rel_size'] + (coef['C'] * group_by_part.loc[i, 'rel_std']))))) * data['Value'].sum() * 0.25
        if len(data) == 1:
            thr = data['Value'].sum() - 0.01
        
        rd.seed(100)
        np.random.seed(100)
        print('Running max-p for part: ' + str(i))
        r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=100)
        print('Number of clusters after max-p: ' + str(r.p))

        if r.p == 0:
            import pdb; pdb.set_trace()
            print('No initial solution found. Removing disconnected areas again.')
            gal = libpysal.open('%d.gal' % i, 'w')
            gal.write(w)
            gal.close()
            gal = libpysal.open('%d.gal' % i, 'r')
            w = gal.read()
            gal.close()
            [n_components, labels] = cg.connected_components(w.sparse)
            print('Disconnected areas exist again')
            for comp in range(n_components):
                import pdb; pdb.set_trace()
                ss = [uu for uu, x in enumerate(labels == comp) if x]
                dd = data.loc[ss]
                dd['F'] = 1
                dd['geometry'] = dd['geometry'].buffer(0)
                dd = dd.dissolve(by='F')
                dd.index = [len(data)]
                dissolve = data.drop(ss)
                dissolve = dissolve.append(dd)
                knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                for cc in range(1, len(data) - 1):
                    countern = 0
                    knn = ps.weights.KNN.from_dataframe(data, k=cc)
                    for s in range(len(ss)):
                        if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                            w.neighbors[str(ss[s])] = w.neighbors[str(ss[s])] + [str(knnw.neighbors[len(data)][0])]
                            w.neighbors[str(knnw.neighbors[len(data)][0])] = w.neighbors[
                                                                                 str(knnw.neighbors[len(data)][0])] + [
                                                                                 str(ss[s])]
                            countern = countern + 1
                            continue
                    if countern > 0:
                        break

            np.random.seed(random_no)
            print('Running max-p again.')
            logger.info('Running max-p again on part: ' + str(i))
            r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
            print('Number of clusters:')
            print(r.p)
            logger.info('Number of clusters after max-p: ' + str(r.p))
        data['CL'] = pd.Series(r.area2region).reindex(data.index)
        data['geometry'] = data['geometry'].buffer(0)
        
        # Calculating the area of each cluster using Lambert Cylindrical Equal Area EPSG:9835 (useful for the density, but needs a projection)
        if param["CRS"] == "epsg:4326":
            data.to_crs("+proj=cea")
            data['area'] = data['geometry'].area / 10**6
            data.to_crs(epsg=4326)
        else:
            data['area'] = data['geometry'].area / 10**6

        # Calculate the aggregated values for each cluster based on the aggregation method
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            if param["agg"][counter_files] == 'density':
                data[raster_name] = data[raster_name] * data['area']
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].sum() / data.loc[data['CL'] == cl, 'area'].sum()
            if param["agg"][counter_files] == 'mean':
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].mean()
            elif param["agg"][counter_files] == 'sum':
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].sum()
        
        file = data.dissolve(by='CL')
        file.reset_index(inplace=True)

        # Result for every part after max-p one
        file.to_file(driver='ESRI Shapefile', filename=paths['parts_max_p'] + 'max_p_part_%d.shp' % i)

    print('Merging all parts of max-p 1')
    gdf = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    gdf['geometry'] = gdf.buffer(0)
    gdf.to_file(driver='ESRI Shapefile', filename=paths['max_p_combined'])

    timecheck("End")


def max_p_algorithm(paths, param):
    """
    This function applies the max-p algorithm to the obtained polygons.
    
    :param param: The parameters from config.py
    :param paths: The paths to the rasters and to the output folders, from config.py
    """
    timecheck("Start")

    # Read all necessary inputs from CSV files for this function.
    df = pd.read_csv(paths["non_empty_rasters"], sep=';', decimal=',', index_col=[0,1])
    non_empty_rasters = list(set(df.index.levels[1].tolist()))
    
    # Group by part number, and calculate the product of rel_size and rel_std
    group_by_part = df.reset_index(inplace=False)
    group_by_part = group_by_part.groupby(['part']).prod()

    # Start max-p
    for i in non_empty_rasters:
        print('Running max-p for part: ' + paths["polygons"] + 'result_%d.shp' % i)
        data = gpd.read_file(paths["polygons"] + 'result_%d.shp' % i)
        
        # Calculate the weighted sum in 'Value', that will be used for clustering
        data['Value'] = 0
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            scaling = data[raster_name].mean()
            data['Value'] = data['Value'] + param["weights"][counter_files] * data[raster_name] / scaling

        # Create weights object
        w = ps.weights.Queen.from_shapefile(paths["polygons"] + 'result_%d.shp' % i)

        # This loop is used to force any disconnected group of polygons to be assigned to the nearest neighbors
        if len(data) > 1:
            knnw = ps.weights.KNN.from_shapefile(paths["polygons"] + 'result_%d.shp' % i, k=1)
            # Attach islands if any to nearest neighbor
            w = libpysal.weights.util.attach_islands(w, knnw)

            [n_components, labels] = cg.connected_components(w.sparse)
            if n_components > 1:
                print('Disconnected areas exist. Removing them before max-p can be applied.')
                for comp in range(n_components):
                    # Filter polygons within that component
                    data_comp = data.loc[labels == comp]
                    
                    import pdb; pdb.set_trace()
                    ss = [uu for uu, x in enumerate(labels == comp) if x]
                    dd = data.loc[ss]
                    dd['F'] = 1
                    dd['geometry'] = dd['geometry'].buffer(0)
                    dd = dd.dissolve(by='F')
                    dd.index = [len(data)]
                    dissolve = data.drop(ss)
                    dissolve = dissolve.append(dd)
                    knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                    for cc in range(1, len(data) - 1):
                        countern = 0
                        knn = ps.weights.KNN.from_dataframe(data, k=cc)
                        for s in range(len(ss)):
                            if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                                w.neighbors[ss[s]] = w.neighbors[ss[s]] + knnw.neighbors[len(data)]
                                w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[
                                                                                knnw.neighbors[len(data)][0]] + [
                                                                                ss[s]]
                                countern = countern + 1
                                continue
                        if countern > 0:
                            break
        # Get coefficients for threshold equation
        coef = get_coefficients(paths)

        # Calculate threshold depending on the size and standard deviation
        thr = (coef['A'] * (exp(-coef['B'] * (group_by_part.loc[i, 'rel_size'] + (coef['C'] * group_by_part.loc[i, 'rel_std']))))) * data['Value'].sum() * 0.25
        if len(data) == 1:
            thr = data['Value'].sum() - 0.01
        
        rd.seed(100)
        np.random.seed(100)
        print('Running max-p for part: ' + str(i))
        r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=100)
        print('Number of clusters after max-p: ' + str(r.p))

        if r.p == 0:
            import pdb; pdb.set_trace()
            print('No initial solution found. Removing disconnected areas again.')
            gal = libpysal.open('%d.gal' % i, 'w')
            gal.write(w)
            gal.close()
            gal = libpysal.open('%d.gal' % i, 'r')
            w = gal.read()
            gal.close()
            [n_components, labels] = cg.connected_components(w.sparse)
            print('Disconnected areas exist again')
            for comp in range(n_components):
                import pdb; pdb.set_trace()
                ss = [uu for uu, x in enumerate(labels == comp) if x]
                dd = data.loc[ss]
                dd['F'] = 1
                dd['geometry'] = dd['geometry'].buffer(0)
                dd = dd.dissolve(by='F')
                dd.index = [len(data)]
                dissolve = data.drop(ss)
                dissolve = dissolve.append(dd)
                knnw = ps.weights.KNN.from_dataframe(dissolve, k=1)
                for cc in range(1, len(data) - 1):
                    countern = 0
                    knn = ps.weights.KNN.from_dataframe(data, k=cc)
                    for s in range(len(ss)):
                        if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                            w.neighbors[str(ss[s])] = w.neighbors[str(ss[s])] + [str(knnw.neighbors[len(data)][0])]
                            w.neighbors[str(knnw.neighbors[len(data)][0])] = w.neighbors[
                                                                                 str(knnw.neighbors[len(data)][0])] + [
                                                                                 str(ss[s])]
                            countern = countern + 1
                            continue
                    if countern > 0:
                        break

            np.random.seed(random_no)
            print('Running max-p again.')
            logger.info('Running max-p again on part: ' + str(i))
            r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
            print('Number of clusters:')
            print(r.p)
            logger.info('Number of clusters after max-p: ' + str(r.p))
        data['CL'] = pd.Series(r.area2region).reindex(data.index)
        data['geometry'] = data['geometry'].buffer(0)
        
        # Calculating the area of each cluster using Lambert Cylindrical Equal Area EPSG:9835 (useful for the density, but needs a projection)
        if param["CRS"] == "epsg:4326":
            data.to_crs("+proj=cea")
            data['area'] = data['geometry'].area / 10**6
            data.to_crs(epsg=4326)
        else:
            data['area'] = data['geometry'].area / 10**6

        # Calculate the aggregated values for each cluster based on the aggregation method
        for counter_files in range(len(paths["inputs"])):
            raster_name = param["raster_names"].split(" - ")[counter_files][:10]
            if param["agg"][counter_files] == 'density':
                data[raster_name] = data[raster_name] * data['area']
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].sum() / data.loc[data['CL'] == cl, 'area'].sum()
            if param["agg"][counter_files] == 'mean':
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].mean()
            elif param["agg"][counter_files] == 'sum':
                for cl in data.loc[pd.notnull(data[raster_name]), 'CL'].unique():
                    data.loc[data['CL'] == cl, raster_name] = data.loc[data['CL'] == cl, raster_name].sum()
        
        file = data.dissolve(by='CL')
        file.reset_index(inplace=True)

        # Result for every part after max-p one
        file.to_file(driver='ESRI Shapefile', filename=paths['parts_max_p'] + 'max_p_part_%d.shp' % i)

    print('Merging all parts of max-p 1')
    gdf = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % non_empty_rasters[0])
    for j in non_empty_rasters[1:]:
        gdf_aux = gpd.read_file(paths['parts_max_p'] + 'max_p_part_%d.shp' % j)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_aux], ignore_index=True))

    gdf['CL'] = gdf.index
    gdf['geometry'] = gdf.buffer(0)
    gdf.to_file(driver='ESRI Shapefile', filename=paths['max_p_combined'])

    timecheck("End")


def max_p_algorithm_2(paths, param):
    """
    This function runs the max-p algorithm again on the results obtained from max_p_algorithm().

    :param param = The parameters from config.py
    :param paths = The paths to the rasters and to the output folders, from config.py
    """

    logger.info('Starting "max_p_algorithm_2".')
    print('------------------------- Max-p Two -------------------------')
    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('This part started at: ' + format_time)
    logger.info('"max_p_algorithm_2" started at: ' + format_time)

    logger.info('Opening file: ' + paths["parts_max_p"] + 'max_p_combined.shp')
    data = gpd.read_file(paths["parts_max_p"] + 'max_p_combined.shp')
    
    # Calculate the weighted sum in 'Value', that will be used for clustering
    counter_files = 'A'
    data['Value'] = 0
    for input_file in paths["inputs"]:
        scaling = data['Value_' + counter_files].mean()
        data['Value'] = data['Value'] + param["weights"][counter_files] * data['Value_' + counter_files] / scaling
        counter_files = chr(ord(counter_files) + 1)

    logger.info('Creating weights object.')
    w = ps.weights.Queen.from_shapefile(paths["parts_max_p"] + 'max_p_combined.shp')
    knnw = ps.weights.KNN.from_shapefile(paths["parts_max_p"] + 'max_p_combined.shp', k=1)
    w = libpysal.weights.util.attach_islands(w, knnw)
    [n_components, labels] = cg.connected_components(w.sparse)
    print(labels)
    aa = w.islands
    if n_components > 1:
        logger.info('Disconnected areas exist. Removing them for max-p.')
        print('Disconnected areas exist')
        for comp in range(n_components):
            import pdb; pdb.set_trace()
            ss = [uu for uu, x in enumerate(labels == comp) if x]
            dd = data.loc[ss]
            dd['F'] = 1
            dd['geometry'] = dd['geometry'].buffer(0)
            dd = dd.dissolve(by='F')
            dd.index = [len(data)]
            Dissolve = data.drop(ss)
            Dissolve = Dissolve.append(dd)
            knnw = ps.weights.KNN.from_dataframe(Dissolve, k=1)
            for cc in range(1, len(data) - 1):
                countern = 0
                knn = ps.weights.KNN.from_dataframe(data, k=cc)
                for s in range(len(ss)):
                    if knn.neighbors[ss[s]][cc - 1] == knnw.neighbors[len(data)][0]:
                        w.neighbors[ss[s]] = w.neighbors[ss[s]] + knnw.neighbors[len(data)]
                        w.neighbors[knnw.neighbors[len(data)][0]] = w.neighbors[knnw.neighbors[len(data)][0]] + [ss[s]]
                        countern = countern + 1
                        continue
                if countern > 0:
                    break
    logger.info('Correcting neighbors.')
    print('Correcting neighbors.')
    w.neighbors = find_neighbors_in_shape_file(paths, w.neighbors)
    print('Neighbors corrected!')
    logger.info('Neighbors corrected.')

    thr = 0.0275 * data['Value'].sum()
    logger.debug('Threshold = ' + str(thr))
    random_no = rd.randint(1000, 1500)  # The range is selected randomly.
    logger.debug('Random number for seed = ' + str(random_no))
    np.random.seed(random_no)

    print('Neighbors Assigned. Running max-p.')
    logger.info('Running max-p.')
    r = ps.region.maxp.Maxp(w, data['Value'].values.reshape(-1, 1), floor=thr, floor_variable=data['Value'], initial=5000)
    print('Max-p finished!')
    print('Number of clusters: ' + str(r.p))
    logger.info('Number of clusters: ' + str(r.p))

    data['CL'] = pd.Series(r.area2region).reindex(data.index)
    data['geometry'] = data.buffer(0)
    if args.type == 'mean':
        output = data.dissolve(by='CL', aggfunc='mean')
    elif args.type == 'sum':
        output = data.dissolve(by='CL', aggfunc='sum')
    output.reset_index(inplace=True)
    output['NAME_0'] = 'CL'
    aux = [str(output.loc[i,'CL']).zfill(2) for i in output.index]
    output['NAME_SHORT'] = output['NAME_0'] + aux
    output.crs = {'init': 'epsg:4326'}
    output = output.to_crs(epsg=3034)
    output['Area'] = output.area / 10 ** 6
    logger.info('Creating final output file: ' + folder_names['final_output'] + 'final_result.shp')
    output.to_file(driver='ESRI Shapefile', filename=folder_names['final_output'] + 'final_result.shp')  # Final file

    current_date_time = datetime.datetime.now()
    format_time = current_date_time.strftime("%Y-%m-%d_%H:%M:%S")
    print('------------------------- == -------------------------')
    print('This part finished at: ' + format_time)
    print('------------------------- == -------------------------')
    logger.info('"max_p_algorithm_2" finished at: ' + format_time)

    return True


def find_neighbors_in_shape_file(paths, existing_neighbors):
    """
    This function finds the neighbors in the shape file. Somehow, max-p cannot figure out the correct neighbors and
    some clusters are physically neighbors but they are not considered as neighbors. This is where this function comes
    in.

    :param folder_names = The names of all the folders created for output.
    :param existing_neighbors = The neighbors matrix that is created by using w and knn. The new neighbors are to be added to this matrix.
    """

    df = gpd.read_file(paths["parts_max_p"] + 'max_p_combined.shp')
    df["NEIGHBORS"] = None
    for index, cluster_number in df.iterrows():
        # get 'not disjoint' countries
        import pdb; pdb.set_trace()
        neighbors = df[~df.geometry.disjoint(cluster_number.geometry.buffer(0.005))].CL.tolist()
        df1 = df
        df1.crs = {'init': 'epsg:4326'}
        df1 = df1.to_crs({'init': 'epsg:32662'})
        df2 = cluster_number.to_frame().T
        df2 = gpd.GeoDataFrame(df2, geometry='geometry')
        df2.crs = {'init': 'epsg:4326'}
        df2 = df2.to_crs({'init': 'epsg:32662'})
        df2.geometry = df2.geometry.buffer(100)  # in m
        test = gpd.overlay(df1, df2, how='intersection')
        test['area'] = test['geometry'].area / 10 ** 6  # in km²
        test = test[test['area'] > 0.01]  # avoids that neighbors share only a point or a very small area
        neighbors2 = test.CL_1.tolist()
        neighbors = neighbors2
        # remove own name from the list
        neighbors = [cl_no for cl_no in neighbors if cluster_number.CL != cl_no]
        # add names of neighbors as NEIGHBORS value
        df.at[index, "NEIGHBORS"] = ','.join(str(n) for n in neighbors)

    # Making the w.neighbors dictionary for replacing it in max_p_algorithm_2.
    neighbors_corrected = dict()
    for index, row in df.iterrows():
        neighbors_for_one = row['NEIGHBORS'].split(',')
        neighbors_int = list()
        for neighbor in neighbors_for_one:
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
    of threshold in max-p algorithm.

    :param paths: The names of all the folders created for output.

    :return coef: The coefficient values for A, B and C returned as a dictionary. EXPECTED STRUCTURE: {'a': 0.556901762222155, 'b': 2.9138975880272286, 'c': 0.6164969722472001}
    """
    # Get coefficients for threshold equation
    ul_point, ur_point, ll_point, lr_point = get_x_y_values(paths)
    est = [1, 1, 1]

    # Run equation solver
    coef = fsolve(eq_solver, est, args=(ll_point, ul_point, ur_point), xtol=0.001)

    coef_dict = {'A': coef[0], 'B': coef[1], 'C': coef[2]}
    return coef_dict


def eq_solver(coef, ll_point, ul_point, ur_point):
    """
    This function serves as the solver to find coefficient values A, B and C for our defined function which is used to
    calculate the threshold.

    :param coef: The coefficients which are calculated
    :param ll_point: Coordinates of lower left point.
    :param ul_point: Coordinates of upper left point.
    :param ur_point: Coordinates of upper right point.

    :return f: Coefficient values for A, B and C in a numpy array. A is f[0], B is f[1] and C is f[2].
    """
    A = coef[0]
    B = coef[1]
    C = coef[2]

    f = np.zeros(3)

    f[0] = (A * (exp(-B * (ll_point[0] + (C * ll_point[1]))))) - 0.5
    #f[1] = (A * (exp(-B * (ul_point[0] + (C * ul_point[1]))))) - 0.1
    f[2] = (A * (exp(-B * (ur_point[0] + (C * ur_point[1]))))) - 0.01

    return f

# geoclustering
A code that clusters high-resolution rasters into regions with homogeneous characteristics.

Authors: Mohammad Youssef Mahfouz, Waleed Sattar Khan, Kais Siala

What is GeoClustering?
GeoClustering is a python-based code which created geographical clusters from high resolution maps. The algorithm uses kmeans++ and max-p regions to create contiguous non-concentrated clusters. GeoClustering works with any high-resolution geographical maps or big data maps with hundreds of millions of pixels.
GeoClustering uses divide and conquer technique. The input mad is divided into smaller square shaped areas. The number of areas is defined by the user. K-means++, and max-p regions are applied respectively to every area. The output areas are then merged together into one map. Finally, max-p regions algorithm is applied to the entire map after merging to get the final output.
K-means++ is used to cluster the data spatially by adding the longitude and latitude of every pixel as a constraint. The number of clusters of k-means++ is decided upon using the elbow method which can be explained in [1]. K-means++ is only used to decrease the resolution with minimum loss of information but can not be used for the entire algorithm as the clusters produced by k-means++ are concentrated clusters having voronoi polygons shape.
After k-means++ decreases the resolution, max-p regions is used in two stages. The first stage is on the square shaped areas, providing non-concentrated clusters on that level. The next stage is done after merging the square shaped areas together to have the entire map. Max-p regions is then applied again to provide the final output.

Input:
High resolution raster file with .tiff extension. GeoClustering does not work with shapefile maps as input. The null values in the input raster should have ONLY Nan or negative values. 
All negative values are assumed to be null values. Therefore, if the input map has a negative value which is not null the code needs editing by the user at line “XX” by changing “0” to the minimum value in the input map.

Requirements:
The main libraries required by GeoClustering are:
-	Pysal
-	libpysal
-	sklearn
-	pandas
-	geopandas
-	numpy
-	fiona
-	shapely

Output:
The output of GeoClustering is a shapefile, with the clusters of the input map as polygons.

References:
[1]	https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera

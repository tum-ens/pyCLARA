# geoclustering
Tool to cluster high-resolution spatial data into contiguous, homogeneous regions

Authors: Kais Siala, Mohammad Youssed Mahfouz, Waleed Sattar Khan

[![Documentation Status](https://readthedocs.org/projects/geoclustering/badge/?version=latest)](http://geoclustering.readthedocs.io/en/latest/?badge=latest)

## What is geoclustering?
geoclustering is a python-based code which creates geographical clusters from high resolution maps. When used with rasters, the algorithm uses kmeans++ and max-p regions to create contiguous non-concentrated clusters. geoclustering works with any high-resolution map with hundreds of millions of pixels.
It uses the divide and conquer technique. The input map is divided into smaller square shaped areas. The number of areas is defined by the user. K-means++, and max-p regions are applied respectively to every area. The output areas are then merged together into one map. Finally, max-p regions algorithm is applied to the entire map after merging to get the final output.
K-means++ is used to cluster the data spatially by adding the longitude and latitude of every pixel as a constraint. The number of clusters of k-means++ is decided upon using the elbow method. K-means++ is only used to decrease the resolution with minimum loss of information but cannot be used for the entire algorithm as the clusters produced by k-means++ are concentrated clusters having shapes similar to voronoi polygons.
After k-means++ decreases the resolution, max-p regions is used in two stages. The first stage is on the square shaped areas, providing non-concentrated clusters on that level. The next stage is done after merging the square shaped areas together to have the entire map. Max-p regions is then applied again to provide the final output.

## Features
* Clustering of one or multiple high-resolution rasters, such as wind resource maps or load density maps
* Supported aggregation functions: average, sum, or density
* Combination of k-means and max-p algorithms, to ensure contiguity
* Clustering of grid data using a hierarchical algorithm
* Flexibility in the number of polygons obtained

## Applications
This code is useful if:

* You want to obtain regions for energy system models with homogeneous characteristics (e.g. similar wind potential)
* You want to cluster regions based on several characteristics simultaneously
* You want to take into account grid restrictions when defining regions for power system modeling

## Related publications

* Siala, Kais; Mahfouz, Mohammad Youssef: [Impact of the choice of regions on energy system models](http://doi.org/https://doi.org/10.1016/j.esr.2019.100362). Energy Strategy Reviews 25, 2019, 75-85.

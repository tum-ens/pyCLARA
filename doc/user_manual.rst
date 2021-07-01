User manual
===========

Installation
------------

.. NOTE:: We assume that you are familiar with `git <https://git-scm.com/downloads>`_ and `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_.

First, clone the git repository in a directory of your choice using a Command Prompt window::

	$ ~\directory-of-my-choice> git clone https://github.com/tum-ens/pyCLARA.git

We recommend using conda and installing the environment from the file ``pyCLARA.yml`` that you can find in the repository. In the Command Prompt window, type::

	$ cd pyCLARA\env\
	$ conda env create -f pyCLARA.yml

Then activate the environment::

	$ conda activate pyCLARA

In the folder ``code``, you will find multiple files:

.. tabularcolumns::
	|l|l|

+-----------------------------------------+-------------------------------------------------------------------------------------+
| File                                    | Description                                                                         |
+=========================================+=====================================================================================+
| config.py                               | used for configuration, see below.                                                  |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| runme.py                                | main file, which will be run later using ``python runme.py``.                       |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\initialization.py                  | used for initialization.                                                            |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\create_subproblems.py              | used to split the clustering problem into smaller subproblems.                      |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\kmeans_functions.py                | includes functions related to the k-means clustering algorithm.                     |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\max_p_functions.py                 | includes functions related to the max-p clustering algorithm.                       |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\lines_clustering_functions.py      | includes functions for the hierarchical clustering of transmission lines.           |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\spatial_functions.py               | contains helping functions for spatial operations.                                  |
+-----------------------------------------+-------------------------------------------------------------------------------------+
| lib\\util.py                            | contains minor helping functions and the necessary python libraries to be imported. |
+-----------------------------------------+-------------------------------------------------------------------------------------+


config.py                                                                                           
---------
This file contains the user preferences, the links to the input files, and the paths where the outputs should be saved.
The paths are initialized in a way that follows a particular folder hierarchy. However, you can change the hierarchy as you wish.

.. toctree::
   :maxdepth: 3
   
   source/config
   
runme.py
--------
``runme.py`` calls the main functions of the code:

.. literalinclude:: ../code/runme.py
   :language: python
   :linenos:
   :emphasize-lines: 12, 19, 20, 23, 26

Recommended input sources
-------------------------
For a list of GIS data sources, check this `wikipedia article <https://en.wikipedia.org/wiki/List_of_GIS_data_sources>`_.

Shapefile of transmission lines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
High-voltage power grid data for Europe and North America can be obtained from `GridKit <https://zenodo.org/record/47317>`_, which used
OpenStreetMap as a primary data source. 

In this repository, the minimum requirements are a shapefile of lines with the attributes ID, Cap_MVA and Type. Such a file
can be obtained using the repository tum-ens/pyPRIMA to clean the GridKit data.

Shapefile of the region of interest (useful for lines clustering)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The shapefile allows to identify the lines that are inside the scope, and those that lie outside of it. Hence, only the outer borders of the polygons matter. If you are interested in administrative divisions, you may consider downloading the shapefiles from 
the website of the Global Administration Divisions `(GADM) <https://gadm.org/download_country_v3.html>`_. You can also create your 
own shapefiles using a GIS software.

High resolution rasters
^^^^^^^^^^^^^^^^^^^^^^^^
Any raster can be used. If multiple rasters are used, ensure that they have the same resolution, the same projection, and the same geographic extent.
Rasters for renewable potentials can be generated using the GitHub repository `tum-ens/pyGRETA <https://github.com/tum-ens/pyGRETA>`_.
A high resolution raster (30 arcsec) of population density can be downloaded from the website of `SEDAC
<https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11/data-download>`_ after registration. A high resolution raster (15 arcsec = 1/240° longitude and 1/240° latitude) made of 24 tiles can be downloaded from `viewfinder panoramas
<http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org15.htm>`_.


Recommended workflow
--------------------
The script is designed to be modular and split into four multiple modules: :mod:`lib.create_subproblems`, :mod:`lib.kmeans_functions`, :mod:`lib.max_p_functions`, and :mod:`lib.lines_clustering_functions`. 

.. WARNING:: The outputs of some modules serve as inputs to the others (this applies to the first three modules). Therefore, the user will have to run the script sequentially.

There are two recommended use cases: either the clustering of rasters, or the clustering of transmission lines. The modules will be presented in the order in which the user will have to run them. 

1. :ref:`clusterRasters`

  a. :ref:`createSubproblems`
  b. :ref:`kmeansFunctions`
  c. :ref:`maxpFunctions`
  
2. :ref:`clusterLines`

It is recommended to thoroughly read through the configuration file `config.py` and modify the input paths and 
computation parameters before starting the `runme.py` script.
Once the configuration file is set, open the `runme.py` file to define what use case you will be using the script for.

.. _clusterRasters:

Clustering of rasters
^^^^^^^^^^^^^^^^^^^^^^
The first use case deals with the clustering of high resolution rasters. Depending on the size of the rasters and the user preferences, all or some of these modules will be used:

.. _createSubproblems:

Creating subproblems
*********************
Instead of applying the clustering algorithms directly on the original rasters, the module :mod:`lib.create_subproblems` is
called to split the rasters into smaller ones. There are two options: either to split the rasters into rectangles of similar sizes,
or according to polygon shapes provided by the user.

.. NOTE:: If you would like to cluster data on a European level, but would like to do it for each country separately, provide a shapefile of European countries to define the subregions.

You can also skip this step altogether and cluster the whole dataset at once (not recommended for very large maps, because
the quality of the results and the speed of the calculation are affected).

.. _kmeansFunctions:

k-means clustering
*******************
This step is also optional. The purpose is to compress the amount of information so that the max-p algorithm can be applied.
The max-p algorithm can cluster up to ~1800 polygons in about 8h. So the functions inside the :mod:`lib.kmeans_functions` module
have two goals: reduce the number of data points while preserving the maximum amount of information that can be processed by the
next module, and polygonizing the rasters (i.e. create shapefiles of polygons from clustered rasters).

The user can define the number of clusters *k*, or let the code decide depending on the standard deviation and the number of valid data points.

.. _maxpFunctions:

max-p functions
****************
The max-p algorithm is the one that ensures the spatial contiguity of the clustered data points, but due to its computational
complexity, the previous steps might be needed. Also, there is a possibility to run the max-p algorithm in two steps: first on
each subproblem separately, then on the whole geographic scope after combining the results of the subproblems. If the number of polygons
after k-means is manageable in one run, then the max-p algorithm is applied once. The result is a shapefile of contiguous polygons
with similar characteristics.

.. NOTE:: You can customize the properties, the weights of the data sets, and the aggregating functions to be used during the clustering, e.g. 20% solar FLH (averaged), 40% wind FLH (averaged), and 40% electricity demand (summed).

.. _clusterLines:

Clustering of transmission lines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This use case is currently independent of the previous one. Starting from the maximum number of regions surrounding each node in the grid,
it uses a hierarchical algorithm to merge connected nodes that have a high transmission capacity flowing from/into them and a low area.
The algorithm stops when the target number of clusters is reached.
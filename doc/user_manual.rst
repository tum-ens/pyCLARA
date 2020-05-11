User manual
===========

Installation
------------

.. NOTE:: We assume that you are familiar with `git <https://git-scm.com/downloads>`_ and `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_.

First, clone the git repository in a directory of your choice using a Command Prompt window::

	$ ~\directory-of-my-choice> git clone https://github.com/tum-ens/pyCLARA.git

We recommend using conda and installing the environment from the file ``geoclustering.yml`` that you can find in the repository. In the Command Prompt window, type::

	$ cd pyCLARA\env\
	$ conda env create -f geoclustering.yml

Then activate the environment::

	$ conda activate geoclustering

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
   :emphasize-lines: 12, 19, 20, 23, 27

Recommended input sources
-------------------------
For a list of GIS data sources, check this `wikipedia article <https://en.wikipedia.org/wiki/List_of_GIS_data_sources>`_.

Shapefile of transmission lines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Shapefile of the region of interest (useful for lines clustering)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The shapefile allows to identify the lines that are inside the scope, and those that lie outside of it. Hence, only the outer borders of the polygons matter. If you are interested in administrative divisions, you may consider downloading the shapefiles from 
the website of the Global Administration Divisions `(GADM) <https://gadm.org/download_country_v3.html>`_. You can also create your 
own shapefiles using a GIS software.



Recommended workflow
--------------------

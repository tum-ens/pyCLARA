Implementation
===============

Start with the configuration:

.. toctree::
   :maxdepth: 3
   
   source/config
   
You can run the code by typing::

	$ python runme.py
	
The script ``runme.py`` calls the main functions of the code, which are explained in the following sections.

.. toctree::
   :maxdepth: 3
   
   source/initialization
   source/create_subproblems
   source/kmeans_functions
   source/max_p_functions
   
For the hierarchical clustering of the transmission network, use the following script:

.. toctree::
   :maxdepth: 3

   source/lines_clustering_functions

Helping functions for the models are included in ``spatial_functions.py``, and ``util.py``.

.. toctree::
   :maxdepth: 3
   
   source/spatial_functions
   source/util
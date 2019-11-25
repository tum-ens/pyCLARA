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
   
For the hierarchical clustering of the transmission network, use the following scripts:

.. toctree::
   :maxdepth: 3

   source/clustering_transmission
   source/connecting_transmission_islands

Helping functions for the models are included in ``helping_functions.py``, ``spatial_functions.py``, and ``util.py``.

.. toctree::
   :maxdepth: 3
   
   source/helping_functions
   source/spatial_functions
   source/util
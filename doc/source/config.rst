Main configuration function
---------------------------

.. automodule:: config
   :members: configuration, general_settings
   
.. NOTE::
   Both *param* and *paths* will be updated in the code after running the function :mod:`config.configuration`.

.. NOTE::
   ``root`` points to the directory that contains all the inputs and outputs.
   All the paths will be defined relatively to the root, which is located in a relative position to the current folder.

.. automodule:: config
   :noindex:
   :members: scope_paths_and_parameters
   
User preferences
----------------

.. automodule:: config
   :noindex:
   :members: computation_parameters, raster_parameters, raster_cutting_parameters, kmeans_parameters, maxp_parameters, transmission_parameters
   
Paths
------

.. automodule:: config
   :noindex:
   :members: output_folders, output_paths
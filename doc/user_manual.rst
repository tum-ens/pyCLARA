User manual
===========

Installation
------------

.. NOTE:: We assume that you are familiar with `git <https://git-scm.com/downloads>`_ and `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_.

First, clone the git repository in a directory of your choice using a Command Prompt window::

	$ ~\directory-of-my-choice> git clone https://github.com/tum-ens/geoclustering.git

We recommend using conda and installing the environment from the file ``geo_clust.yml`` that you can find in the repository. In the Command Prompt window, type::

	$ cd renewable-timeseries\env\
	$ conda env create -f geo_clust.yml

Then activate the environment::

	$ conda activate geo_clust

In the folder ``code``, you will find multiple files:

.. tabularcolumns::
	|l|l|

+---------------------------------------+----------------------------------------------------------------------------------+
| File                                  | Description                                                                      |
+=======================================+==================================================================================+
| config.py                             | used for configuration, see below.                                               |
+---------------------------------------+----------------------------------------------------------------------------------+


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

missing


Recommended input sources
-------------------------


Recommended workflow
--------------------

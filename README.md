# geoclustering
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)
[![Documentation Status](https://readthedocs.org/projects/geoclustering/badge/?version=latest)](http://geoclustering.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

Tool to cluster high-resolution spatial data into contiguous, homogeneous regions

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

* Siala, Kais; Mahfouz, Mohammad Youssef: [Impact of the choice of regions on energy system models](https://doi.org/10.1016/j.esr.2019.100362). Energy Strategy Reviews 25, 2019, 75-85.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/kais-siala"><img src="https://avatars2.githubusercontent.com/u/21306297?v=4" width="100px;" alt="kais-siala"/><br /><sub><b>kais-siala</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=kais-siala" title="Code">💻</a> <a href="https://github.com/tum-ens/geoclustering/commits?author=kais-siala" title="Documentation">📖</a> <a href="#example-kais-siala" title="Examples">💡</a> <a href="#ideas-kais-siala" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-kais-siala" title="Maintenance">🚧</a> <a href="#review-kais-siala" title="Reviewed Pull Requests">👀</a> <a href="#talk-kais-siala" title="Talks">📢</a></td>
    <td align="center"><a href="https://github.com/waleedskhan"><img src="https://avatars0.githubusercontent.com/u/48930932?v=4" width="100px;" alt="Waleed Sattar Khan"/><br /><sub><b>Waleed Sattar Khan</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=waleedskhan" title="Code">💻</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
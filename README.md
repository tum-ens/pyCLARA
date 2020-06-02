# pyCLARA
[![Documentation Status](https://readthedocs.org/projects/geoclustering/badge/?version=latest)](http://geoclustering.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/175087742.svg)](https://zenodo.org/badge/latestdoi/175087742)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)

**py**thon **C**lustering of **L**ines **A**nd **RA**sters: a tool to cluster high-resolution spatial data (rasters or polylines connecting Voronoi polygons) into contiguous, homogeneous regions.

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
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/kais-siala"><img src="https://avatars2.githubusercontent.com/u/21306297?v=4" width="100px;" alt=""/><br /><sub><b>kais-siala</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=kais-siala" title="Code">💻</a> <a href="https://github.com/tum-ens/geoclustering/commits?author=kais-siala" title="Documentation">📖</a> <a href="#example-kais-siala" title="Examples">💡</a> <a href="#ideas-kais-siala" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-kais-siala" title="Maintenance">🚧</a> <a href="https://github.com/tum-ens/geoclustering/pulls?q=is%3Apr+reviewed-by%3Akais-siala" title="Reviewed Pull Requests">👀</a> <a href="#talk-kais-siala" title="Talks">📢</a></td>
    <td align="center"><a href="https://github.com/HoussameH"><img src="https://avatars2.githubusercontent.com/u/48953960?v=4" width="100px;" alt=""/><br /><sub><b>HoussameH</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=HoussameH" title="Code">💻</a> <a href="https://github.com/tum-ens/geoclustering/commits?author=HoussameH" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/waleedskhan"><img src="https://avatars0.githubusercontent.com/u/48930932?v=4" width="100px;" alt=""/><br /><sub><b>Waleed Sattar Khan</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=waleedskhan" title="Code">💻</a> <a href="https://github.com/tum-ens/geoclustering/commits?author=waleedskhan" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/MYMahfouz"><img src="https://avatars3.githubusercontent.com/u/33868271?v=4" width="100px;" alt=""/><br /><sub><b>MYMahfouz</b></sub></a><br /><a href="https://github.com/tum-ens/geoclustering/commits?author=MYMahfouz" title="Code">💻</a> <a href="#ideas-MYMahfouz" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Please cite as
We prefer that you cite the original publication, where the tool was introduced:
* Siala, Kais; Mahfouz, Mohammad Youssef: [Impact of the choice of regions on energy system models](https://doi.org/10.1016/j.esr.2019.100362). Energy Strategy Reviews 25, 2019, 75-85.
If you are using a new feature that was not part of the publication, and would like to cite it:
* Kais Siala, Houssame Houmy, Waleed Sattar Khan, & Mohammad Youssef Mahfouz. (2020, June 1). tum-ens/pyCLARA: python Clustering of Lines And RAsters (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3872274

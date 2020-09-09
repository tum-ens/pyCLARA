from lib.initialization import initialization
from lib.create_subproblems import cut_raster
from lib.kmeans_functions import *
from lib.max_p_functions import *
from lib.lines_clustering_functions import *
from lib.util import *

if __name__ == "__main__":

    paths, param = initialization()

    cut_raster(paths, param)

    # # k-means functions
    calculate_stats_for_non_empty_rasters(paths, param)
    if param["kmeans"]["method"] == "reference_part":
        choose_ref_part(paths)
        identify_max_number_of_clusters_in_ref_part(paths, param)
    k_means_clustering(paths, param)
    polygonize_after_k_means(paths, param)

    # max-p functions
    max_p_clustering(paths, param)

    # lines clustering functions
    # lines_clustering(paths, param)
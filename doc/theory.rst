*******
Theory
*******
Introduction
=============
A number of different algorithms have been developed by scientists for clustering geographic data. The most trivial and commonly used algorithm is the k-means algorithm but it has some limitations. These limitations urged the researchers to come up with more novel and sophisticated methods of clustering geographic data. Usually, the type of algorithm which is used depends on the type of data available and the associated time complexity and requirements for clustering quality. Furthermore, the clustering algorithms are based on different techniques and one of these techniques is hierarchical clustering.

Hierarchical Clustering
=======================
A hierarchical clustering algorithm forms a dendrogram which is basically a tree that splits the input data into small subsets using a recursive approach [1]. The dendrogram can be built using two different approaches which are “bottom-up” or “top-down” approach. As the name suggests, the bottom-up approach treats each object as a separate cluster and combines the two “closest” clusters into one. The algorithm continues until all objects are merged together to form one big object. On the other hand, the top-down approach starts with one big cluster in which all the objects exist. The big cluster is repeatedly broken down into smaller clusters in each iteration of the algorithm [1]. 

k-means Method
==============
The k-means method is one of the oldest methods that has been used for clustering. In this method, the average value of data objects in a cluster is used as the cluster center [2]. The steps that the k-means method follows are:

# The number of clusters k are chosen. The choice of right k is important as small k can result in clusters in which distances to centroid are very long whereas a too large k means that the algorithm will take longer to converge with little to no improvement in average distances.
# Each cluster is initialized by either of the two methods namely Forgy method or Random partition method. For standard k-means algorithm, Forgy method is preferable in which k random observations are chosen as initial means for the clusters [3].
# This step is known as the assignment step. A total of k number of clusters are created with each data object being assigned to the nearest mean. The figure so obtained is known as Voronoi diagram. 
# The mean of all the data points in each cluster is calculated and the newly calculated mean serves as the new centroid for each cluster. This step is called the update step.
# Step (3) and (4) are repeated until convergence criteria has been attained. Usually, the convergence is achieved when the object assignments do not change any more.

Generally, the k-means algorithm has a time complexity equal to O (n2) [4]. Moreover, it is a heuristic method which means that convergence to a global optimum is not guaranteed. However, as k-means is a relatively fast-algorithm, it is a common practice to run k-means method of clustering with different starting conditions and choose the run that has the best results.

k-means++ Method
==================
The k-means++ clustering algorithm can be thought of as an add-on to the standard k-means clustering algorithm. The k-means++ algorithm is designed to improve the quality of clusters that are formed as local optimum using the standard k-means method [4]. Moreover, this method also tends to lower the runtime of the subsequently used k-means algorithm by making it converge quickly. k-means++ achieves this by choosing “seeds” for k-means algorithm. The “seeds” are the initial k cluster centers which are chosen based on a certain criteria instead of randomly choosing the cluster centers as was the case in the standard k-means algorithm.
The steps that k-means++ algorithm follows for initializing the cluster centers are as follows:

# Choose one data point at random from the given data set and consider it as the first center c¬i.
# Calculate the distance of each of the data points in the set from the nearest cluster center with D(x) being the distance of each point.
# Choose a new cluster center ci.
# Repeat steps (2) and (3) until a k number of centers have been chosen.
# Using the initialized centers, proceed with the standard k-means algorithm as outlined in previous section.
	
Although initializing cluster centers using k-means++ method is computationally expensive, the subsequent k-means method actually takes less time to converge and as a result, the overall time taken to converge is usually less as compared to the standard k-means method. Moreover, the k-means++ method is O (log k) competitive [4]. Furthermore, as mentioned above, the local optimum obtained is much more optimized and therefore, the k-means++ method is better than the standard k-means method for clustering of data. Fig. 3 [5] shows the results of a clustering performed on a set of typical data using k-means and k-means++ method. It is clear from the figure that error is smaller when k-means++ is used for clustering and the convergence time is also almost the same.

max-p Method
=============
The max-p regions problem has been developed to satisfy spatial contiguity constraints while clustering. As opposed to other constraint-based methods, this methods models the total number of sectors as an internal parameter  [6]. Moreover, this approach does not put restrictions on the geometry of regions. In fact, it lets the data sets dictate the shape. The heuristic solution that has been proposed in order to solve the max-p regions problem follows the following steps.

# The first is the construction phase which is subdivided into two more phases. The first sub-phase is named growing phase in which the algorithm selects an unassigned area at random. This is called the seed area.
# The neighboring regions of seed area are continuously added to the first seed until a threshold is reached.
# Then the algorithm chooses a new seed area and grows a region by following step (2). This step is repeated until no new seeds which can satisfy the threshold value can be found.
# Areas which are not assigned to a region are called enclaves. Therefore, at the end of growing phase, a set of enclaves and growing regions are formed.
# The number of growing regions and enclaves can change in successive iterations. Solutions in which maximum growing regions are equal to maximum growing regions in the previous iteration are only kept.
# The next step is the process of enclave assignment and each solution found in previous steps is passed to this step. Each enclave is assigned to a neighboring region based on similarity. This marks the end of construction phase.
# In the next step i.e. the local search phase, a set of neighboring solutions is obtained by modifying the found solutions and improving them. Neighboring solutions are created in a way that each solution is feasible. One way to create neighboring solutions is to move one region to another region  [7].
# Finally, a local search algorithm such as Simulated Annealing, Greedy Algorithm or Tabu Search Algorithm is used to improve the solution.

The main purpose of max-p regions problem and the proposed solution is to aggregate small areas into homogeneous regions in a way that the value of a spatially extensive attribute is always more than the threshold value. It can be useful for clustering rasters in which constraints such as population density, energy per capita need to be met [6].



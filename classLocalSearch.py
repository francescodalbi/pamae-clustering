from sklearn.utils.extmath import stable_cumsum
from sklearn_extra.cluster import KMedoids
import numpy as np

"""
classLocalSearch which extends the KMedoids class from the sklearn_extra.cluster module. 
This class  has some additional initialization parameters and overrides the _kpp_init method of the KMedoids class.

The _kpp_init method is a helper function for initializing the medoids in the k-medoids clustering algorithm. 
It uses a method similar to k-means++ to choose initial seeds for the medoids. 


Best_medoids is an optional parameter of the classLocalSearch class that allows you to specify the initial medoids 
to be used for K-medoid clustering initialization. 
If the parameter is passed, the algorithm will use the specified medoids for initialization, 
otherwise it will use a default initialization method.

The best_medoids parameter is used in the _kpp_init method to initialize the medoids with the specified values. 
Specifically, after setting the number of local seeding attempts, the method initializes the centers
with the best medoids (best_medoids.copy()), that means that after setting the number of local seeding attempts, 
the K-Medoids clustering method randomly selects a set of initial centers. 
Then, instead of using these random centers to start the clustering algorithm, the method replaces the initial centers
with the best medoids, which are the points in the dataset that minimize the sum of the distances to the other points 
in the cluster.
 
The algorithm goes on calculating the list of closest distances and the current 
potential for each cluster (Error Sum of Squares or SSE).
It then internally updates each cluster using the specified medoids as a starting point.

In this way, best_medoids provides an initial solution to the clustering problem, which can lead to faster convergence 
and a better final solution
"""

class KMeoids_localsearch(KMedoids):

    def __init__(
            self,
            n_clusters=8,
            metric="euiclidean",
            method="alternate",
            init="heuristic",
            best_medoids=None,
            max_iter=300,
            random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_medoids = best_medoids

    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):

        # Get the number of points from the distance matrix
        n_points, _ = D.shape

        # Create an empty array to hold the final medoids
        final_medoids = np.empty(n_clusters, dtype=int)

        # Initialize with best medoids
        final_medoids = self.best_medoids.copy()

        # Initialize medoids with those found previously.
        # Calculate the list of closest distances and the current potential for each cluster
        closest_distances_sq = np.zeros((n_points,))
        current_pot = 0

        for i, center_id in enumerate(final_medoids):
            # Calculate distances from center_id
            distances = D[center_id, :]
            # Get the square of distances
            closest_distances_sq_i = distances ** 2
            # Calculate the current potential for this cluster
            current_pot_i = closest_distances_sq_i.sum()
            # Update the list of closest distances and the current potential
            closest_distances_sq[distances < closest_distances_sq_i] = distances[distances < closest_distances_sq_i] ** 2
            current_pot += current_pot_i

        # For each initialized centroid, calculate the list of samples from the centroid to each point and the
        # list of the nearest samples. Update the cluster potential with the sum of the squared samples of the nearest points.
        for center_index, center_id in enumerate(final_medoids):
            # Get the indices of points in the current cluster
            #this is a boolean array of length n_points that indicates which points belong to the current cluster
            _indices = (labels == center_index)
            # Get the distances between points in the current cluster
            _distances = D[_indices][:, _indices]
            # Calculate the current potential for this cluster
            cls_pot = (_distances ** 2).sum()

            """ 
            Get the candidate ids.
            The combination of these two objects selects only the candidate indices that belong to the current cluster, 
            ignoring the indices of other points. 
            This is done by selecting the elements of np.arange(n_points) that correspond 
            to the indices where _indices is True. 
            The result is an array of candidate indices  that belong to the current cluster.
            """
            candidate_ids = np.arange(n_points)[_indices]
            # Calculate the number of local trials for this cluster
            n_local_trials = max(int(2 * np.log(n_clusters)) ** 2, 1)

            # For each cluster, the internal update phase begins.
            for _trial in range(n_local_trials):
                """
                Generate a random value, cls_plot is the current potential for the cluster
                Random sample's range is 0-1
                A candidate point within a cluster is randomly chosen, where the probability of selecting 
                a given point depends on its contribution to the overall cluster potential. 
                In particular, points that contribute more to the cluster potential will have a higher probability 
                of being selected.
                """
                rand_values = random_state_.random_sample() * cls_pot
                # Choose a candidate index based on the random value
                """
                To do this, two operations are performed:
                np.cumsum(_distances.sum(axis=1)): this operation calculates the cumulative sum of the distances between 
                    points in the cluster, along the axis of the rows. This means that you get an array in which 
                    each element corresponds to the sum of the distances between the current point and 
                    all previous points within the cluster.
                np.searchsorted(): this function performs a binary search on the array of cumulative sums of distances 
                    in order to find the index of the first element that exceeds the randomly generated rand_values. 
                    This index corresponds to the index of the selected candidate point.
                """
                candidate_index = np.searchsorted(np.cumsum(_distances.sum(axis=1)), rand_values)
                # Get the candidate id
                candidate_id = candidate_ids[candidate_index]
                # Calculate the new potential if the centroid were replaced by the chosen point
                new_pot = ((D[candidate_id, _indices]) ** 2).sum()

                # If the new potential is smaller, replace the centroid with the chosen point and update the distances
                if new_pot < cls_pot:
                    final_medoids[center_index] = candidate_id
                    """
                    The row _distances[candidate_index] = D[candidate_id, _indices] is used to replace the row of the 
                        current medoid with the chosen candidate point, that is, it replaces the row of 
                        the distance matrix _distances related to the current medoid with the row of the 
                        distance matrix D related to the candidate point.

                    """
                    _distances[candidate_index] = D[candidate_id, _indices]
                    #_distances[:, candidate_index] = D[_indices, candidate_id]
                    cls_pot = new_pot

        # Return the final medoids
        return final_medoids







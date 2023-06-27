from sklearn_extra.cluster import KMedoids
import numpy as np

"""
classLocalSearch which extends the KMedoids class from the sklearn_extra.cluster module. 
This class  has some additional initialization parameters and overrides the _kpp_init method of the KMedoids class.

The _kpp_init method is a helper function for initializing the medoids in the k-medoids clustering algorithm. 

Best_medoids is an optional parameter of the classLocalSearch class that allows you to specify the initial medoids 
to be used for K-medoid clustering initialization. 
If the parameter is passed, the algorithm will use the specified medoids for initialization, 
otherwise it will use a default initialization method.

The best_medoids parameter is used in the _kpp_init method to initialize the medoids with the specified values. 

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
        n_points, _ = D.shape
        # Initialize final_medoids with the best_medoids from previous iterations
        final_medoids = self.best_medoids.copy()

        # Iterate over each center index and center id
        for center_index, center_id in enumerate(final_medoids):
            # Find indices and distances of points belonging to the current center/medoid
            _indices = (labels == center_id)
            _distances = D[_indices][:, _indices]

            # Calculate the potential/clustering error of the current cluster
            cls_pot = (_distances ** 2).sum()

            # Get candidate index and set the number of local trials
            candidate_ids = np.arange(n_points)[_indices]
            n_local_trials = max(int(2 * np.log(n_clusters)) ** 2, 1)

            # Get internal indices for considering only internally driven updates
            internal_indices = np.where(_indices)[0]

            # Perform local trials
            for _trial in range(n_local_trials):
                # Generate a random value within the cluster potential
                rand_values = random_state_.random_sample() * cls_pot

                # Select a candidate index based on cumulative distances
                candidate_index = np.searchsorted(np.cumsum(_distances.sum(axis=1)), rand_values)
                candidate_id = candidate_ids[candidate_index]

                # Calculate the potential of the candidate medoid
                new_pot = ((D[candidate_id, _indices]) ** 2).sum()

                # If the candidate index is an internal index
                if candidate_index in internal_indices:
                    # If the new potential is lower, update the medoid and distances
                    if new_pot < cls_pot:
                        final_medoids[center_index] = candidate_id
                        _distances[candidate_index] = D[candidate_id, _indices]
                        cls_pot = new_pot

        return final_medoids










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
        n_points, _ = D.shape
        final_medoids = np.empty(n_clusters, dtype=int)
        final_medoids = self.best_medoids.copy()

        for center_index, center_id in enumerate(final_medoids):
            _indices = (labels == center_id)
            _distances = D[_indices][:, _indices]
            cls_pot = (_distances ** 2).sum()

            candidate_ids = np.arange(n_points)[_indices]
            n_local_trials = max(int(2 * np.log(n_clusters)) ** 2, 1)

            internal_indices = np.where(_indices)[0]

            for _trial in range(n_local_trials):
                rand_values = random_state_.random_sample() * cls_pot

                candidate_index = np.searchsorted(np.cumsum(_distances.sum(axis=1)), rand_values)
                candidate_id = candidate_ids[candidate_index]
                new_pot = ((D[candidate_id, _indices]) ** 2).sum()

                if candidate_index in internal_indices:
                    if new_pot < cls_pot:
                        final_medoids[center_index] = candidate_id
                        _distances[candidate_index] = D[candidate_id, _indices]
                        cls_pot = new_pot

        return final_medoids









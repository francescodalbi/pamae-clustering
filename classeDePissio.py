from sklearn.utils.extmath import stable_cumsum
from sklearn_extra.cluster import KMedoids
import numpy as np

class ClasseDePissio(KMedoids):

    def __init__(
            self,
            n_clusters=8,
            metric="euclidean",
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


 # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        # Initialize with best medoids
        centers = self.best_medoids.copy()

        # Inizializza i medoidi con quelli trovati precedentemente

        # Initialize list of closest distances and calculate current potential for each cluster
        closest_dist_sq = np.zeros((n_samples,))
        current_pot = 0

        # Inizializza la lista delle sample più vicine per ogni punto, e calcola il potenziale attuale per ogni cluster
        for i, center_id in enumerate(centers):
            distances = D[center_id, :]
            closest_dist_sq_i = distances ** 2
            current_pot_i = closest_dist_sq_i.sum()
            closest_dist_sq[distances < closest_dist_sq_i] = distances[distances < closest_dist_sq_i] ** 2
            current_pot += current_pot_i

            # Per ogni centroide inizializzato, calcola la lista delle sample dal centroide a ogni punto e la lista delle sample più vicine.
            # Inoltre, aggiorna il potenziale del cluster con la somma delle sample quadrate dei punti più vicini.

        # Update each cluster internally
        for center_index, center_id in enumerate(centers):
            cluster_indices = (labels == center_index)
            cluster_distances = D[cluster_indices][:, cluster_indices]
            cluster_pot = (cluster_distances ** 2).sum()
            candidate_ids = np.arange(n_samples)[cluster_indices]
            n_local_trials = max(int(2 * np.log(n_clusters)) ** 2, 1)

            # Per ogni cluster, inizia la fase di aggiornamento interno
            # Inizializza l'indice dei punti nel cluster, le sample tra i punti e il potenziale del cluster.

            for trial in range(n_local_trials):
                rand_vals = random_state_.random_sample() * cluster_pot
                candidate_index = np.searchsorted(np.cumsum(cluster_distances.sum(axis=1)), rand_vals)
                candidate_id = candidate_ids[candidate_index]
                new_pot = ((D[candidate_id, cluster_indices]) ** 2).sum()

                # Per ogni tentativo di aggiornamento interno, scegli casualmente un punto nel cluster e calcola la nuova distanza dal centroide
                # se venisse sostituito dal punto scelto.
                # Se il potenziale del cluster diminuisce, sostituisci il centroide con il punto scelto e aggiorna le sample tra i punti.

                if new_pot < cluster_pot:
                    centers[center_index] = candidate_id
                    cluster_distances[candidate_index] = D[candidate_id, cluster_indices]
                    cluster_distances[:, candidate_index] = D[cluster_indices, candidate_id]
                    cluster_pot = new_pot

        # Ritorna i nuovi centri dei cluster aggiornati internamente
        return centers


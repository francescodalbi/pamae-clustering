from typing import Iterable, Tuple, NamedTuple, List
from collections.abc import Sequence

import random
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession
from sklearn.metrics.pairwise import manhattan_distances
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

import itertools

# Create SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PAMAE") \
    .getOrCreate()

sc = spark.sparkContext
bin = 3
ds_import: ps.RDD[np.ndarray[float]] = sc.textFile("datasets/google_review_ratings_2columns_150rows.csv").map(lambda line: line.split(",")).map(
    lambda x: to_float_conversion(x)
)


def to_float_conversion(l: List[str]) -> np.ndarray[float]:
    float_lst = np.float_(l)
    return float_lst

print(type(ds_import.collect()[0]))
print(ds_import.collect()[0])

def distributed_sampling_and_global_search(
        dataset: ps.RDD[np.ndarray[float]],
        n_bins: int,
        sample_size: int,
        t: int
) -> np.ndarray:
    """
    Random samples are generated by calling the "get_random_samples" function, and in them all possible groups of
    medoids are calculated (according to the "global search" logic) and then the best ones are chosen for each sample
    :param dataset: original dataset
    :param n_bins: number of parts in which to divide the dataset
    :param sample_size: first n elements from each bin/partition of the original dataset
    :param t: desired number of clusters????
    :return:
    """

    samples = get_random_samples(dataset, m=n_bins, n=sample_size)
    print(samples.collect())
    best_medoids = global_search(samples, t)
    refinement(best_medoids, dataset, t)
class Sample(NamedTuple):
    key: int
    rows: np.ndarray[np.ndarray[float]]


class Bin(Sample):
    """
    Creo una classe per definire un tipo di dato che uso spesso.
    In questo caso dato che Bin è lo stesso tipo di dato di Sample, lo passo come parametro e Bin eredita le proprietà
    di Sample.
    """
    pass


def get_random_samples(dataset: ps.RDD[np.ndarray[float]], m: int, n: int) -> ps.RDD[Sample]:
    """
    Random samples are generated by dividing the entire data set into m parts and taking the first n elements of each
    :param dataset: original dataset
    :param m: number of parts in which to divide the dataset
    :param n: first n elements from each bin/partition of the original dataset
    :return: RDD that contains all samples, grouped according to the key generated by the "random_mod" function.
    """

    # TODO Genere chiave rnd, calcolo il modulo e lo assegno alle tuple
    # TODO verificare se "Tuple[int, np.ndarray[float]]" si possa riorganizzare come un oggetto tipo "dato" visto che è
    # TODO sempre uguale

    def random_mod(row: np.ndarray[float]) -> Tuple[int, np.ndarray[float]]:
        # m = numero di campioni desiderati
        rnd = random.randrange(0, 9)
        mod = rnd % m
        return mod, row

    ds_with_mod: ps.RDD[Tuple[int, np.ndarray[float]]] = dataset.map(lambda row: random_mod(row))
    print(sorted(ds_with_mod.groupByKey().mapValues(len).collect()))

    # TODO List è un "interfaccia" che mi permette di generalizzare i tipi di lista suggeriti in input
    # https://docs.python.org/3/library/typing.html
    ds_grouped: ps.RDD[Tuple[int, np.ndarray[np.ndarray[float]]]] = sc.parallelize(
        ds_with_mod.groupByKey().mapValues(list).collect())

    # TODO: capire perchè questa funzione fa esplodere tutto, è un mistero
    def get_first_n_of_bin(bin_: Tuple[int, np.ndarray[np.ndarray[float]]]) -> Sample:
       return Sample(key=bin_[0], rows=bin_[1][:n])

    ds_samples = ds_grouped.map(lambda row: (row[0], row[1][:n]))
    return ds_samples


class SearchResult(NamedTuple):
    medoid_ids: np.ndarray
    """
    Collection of ids that allow to identify the medoids in the full dataset
    """

    total_error: float
    """
    Sum of the errors (distances of each object from the medoid) of 
    all the clusters (identified by each medoid)
    """


def global_search(sample: ps.RDD[np.ndarray[float]], t: int) -> SearchResult:
    """
    Phase I except for the sampling part

    :param sample: collection of objects (rows) from the dataset
    :param t: number of cluster (medoids)
    :return: 1D array of k elements containing the indexes of the best medoids in
        the provided sample
    """

    # 0. Otteniamo tutte le combinazioni C di k medoidi sul sample intero:
    #   C = [(m_0, ..., m_k), ...], dove ogni elemento m_i è l'indice del medoide all'interno del sample
    # 1. selezioniamo k medoidi (una combinazione di quelle ottenute al passo 0.)
    # 2. calcoliamo la matrice A di distanze di Manthattan (A=all_distances())
    #   "2D matrix where element ij is the distance between object (row) i and object (row) j"
    # 3. Calcoliamo i cluster: ottengo una lista [c_1, ..., c_k], dove c_i è
    #       la lista di oggetti (righe) che compone il cluster i
    #   3.0. sappiamo che:
    #       - A è la matrice di distanze all_distances()
    #       - m_0, ..., m_k sono (gli indici dei) medoidi -> es. riga_medoide_2 = sample[m_2]
    #       - p è (l'indice del) punto da classificare
    #   3.1. label_p = argmin(np.array(A[m_0, p], A[m_1, p], ..., A[m_n, p])):
    #       il cluster (label/etichetta) di p è identificato dal medoide m_i la cui distanza da p è minima
    #   3.2. Eseguo dunque il passo 3.1. per ogni indice p del sample e ottengo la lista descritta al punto 3.
    # 4. Calcoliamo l'errore di ogni cluster (somma distanze da medoide) e poi
    #   li sommiamo per ottenere l'errore totale della combinazioni di medoidi
    # 5. Salviamo l'errore per la combinazione corrente e:
    #   5.1. Se ho esaurito le combinazioni, scelgo quella dall'errore minore e la ritorno
    #   5.2. Se NON le ho esaurite, torno al punto 1

    #Passo 0

    # definisci una funzione che elabora una singola riga del RDD

    def process_row(campione):
        print("CAMPIONE: ", campione)
        key = campione[0]
        values = campione[1]
        print("VALUES TYPE: ", type(values))
        print("VALUES : ", values)
        return clustering(values,t, None, key)
    # applica la funzione ad ogni riga dell'RDD sample
    rdd3 = sample.map(process_row)
    # Inizializza la matrice degli errori e dei medoidi
    # inizializza l'array degli errori
    errori = np.empty([0, 3])

    # stampa i risultati
    result = rdd3.collect()

    #STAMPO I RISULTATI
    for key, value in result:
        print(f"Campione {key}:")
        for i, cluster in enumerate(value['clusters']):
            print(f"Cluster {i}:")
            for point in cluster:
                print(point)
        print(f"Medoidi:")
        for medoid in value['medoids']:
            print(medoid)

        # aggiungi l'errore e i medoidi all'array degli errori
        errori = np.append(errori, np.array([medoid[0], medoid[1], value['error']]).reshape(1, -1), axis=0)
        print(f"Errore di clustering: {value['error']}")
        print()

    # ordina gli errori in ordine crescente di valore
    errori_ord = errori[errori[:, 2].argsort()]

    # stampa il set di medoidi con l'errore minimo
    print(f"Set di medoidi migliori: {errori_ord[0, 0:2]}")
    print(f"Errore minimo: {errori_ord[0, 2]}")

    best_medoids = errori_ord[0, 0:2]


    #PLOT dei risultati
    import matplotlib.pyplot as plt

    for key, value in result:
        plt.figure()
        for i, cluster in enumerate(value['clusters']):
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i}")
            medoid = np.array(value['medoids'][i])
            plt.scatter(medoid[0], medoid[1], marker='x', s=200, linewidths=3, color='r')
        plt.legend()
        plt.title(f"Campione {key}")
        plt.show()
    return best_medoids
def refinement(best_medoids: np.ndarray, dataset: ps.RDD, t:int) -> np.ndarray:
    """
    Phase 2 of the algorithm presented in the PAMAE paper
    :param best_medoids: collection of the (ids of the) best medoids found in phase 1
    :param dataset: full dataset
    :param t: number of clusters
    :return: array containing k clusters, each of which is represented by collection of data points
    """
    # 1. Identificare i cluster rispetto ai best_medoids e al dataset intero (punto 2. + 3. di global_search())
    # 2. Per ogni cluster_i. esegui global_search(sample=cluster_i, k=1)
    # 3. Con i medoidi ottenuti al punto 2., calcolo i cluster definitivi

    def process_row(row):
        values = row[1]
        # Converti la lista in un numpy array
        #values = [np.array(row)]
        print("VALUES TYPE: ", type(row))
        print("VALUES: ", row)
        return clustering(row, t, best_medoids)

    rdd_array = sc.parallelize(dataset.map(lambda x: [x]).collect())
    rdd_array_flat = rdd_array.flatMap(lambda x: x)
    print(rdd_array_flat.collect())

    rdd_refinement = rdd_array_flat.map(process_row)
    # inizializza l'array degli errori
    errori = np.empty([0, 3])

    # stampa i risultati
    result = rdd_refinement.collect()

""""
    # STAMPO I RISULTATI
    for key, value in result:
        print(f"Campione {key}:")
        for i, cluster in enumerate(value['clusters']):
            print(f"Cluster {i}:")
            for point in cluster:
                print(point)
        print(f"Medoidi:")
        for medoid in value['medoids']:
            print(medoid)

        # aggiungi l'errore e i medoidi all'array degli errori
        errori = np.append(errori, np.array([medoid[0], medoid[1], value['error']]).reshape(1, -1), axis=0)
        print(f"Errore di clustering: {value['error']}")
        print()

    # ordina gli errori in ordine crescente di valore
    errori_ord = errori[errori[:, 2].argsort()]

    # stampa il set di medoidi con l'errore minimo
    print(f"Set di medoidi migliori: {errori_ord[0, 0:2]}")
    print(f"Errore minimo: {errori_ord[0, 2]}")

    best_medoids = errori_ord[0, 0:2]

    # PLOT dei risultati

    for key, value in result:
        plt.figure()
    for i, cluster in enumerate(value['clusters']):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i}")
        medoid = np.array(value['medoids'][i])
        plt.scatter(medoid[0], medoid[1], marker='x', s=200, linewidths=3, color='r')
        plt.legend()
        plt.title(f"Campione {key}")
        plt.show()
"""



def clustering(distanze: list, t:int, best_medoids = None, key: int = None, ):
    """

    :param distanze: values from samples
    :param t: number of clusters
    :param best_medoids: set of best medoids from the  phase 1 (optional argument)
    :param key: the key that identify each sample (optional argument)
    :return:
    """
    def distances(values: list) -> np.ndarray:
        """
            >>> all_distances([[1, 2], [3, 4]])
            >>> np.ndarray([\
                [0., 4.],\
                [4., 0.]\
                ])

            :param sample: set of objects (dataset rows) sampled from the full dataset
            :return: 2D matrix where element ij is the distance between object (row) i and object (row) j
            """
        data = np.array(values)
        return manhattan_distances(data, data)

    # Calcolo la matrice di distanza
    distance_matrix = distances(distanze)

    # Creo l'istanza del modello KMedoids
    if best_medoids is None:
        kmedoids = KMedoids(n_clusters=t, metric='precomputed')
    else:
        kmedoids = KMedoids(n_clusters=t, metric='precomputed', init='build', max_iter=0)

    # Eseguo il clustering
    kmedoids.fit(distance_matrix)

    # Recupero i medoidi
    medoids_idx = kmedoids.medoid_indices_
    medoids = [distanze[idx] for idx in medoids_idx]

    # Calcolo l'errore di clustering
    labels = kmedoids.labels_
    error = 0
    for i in range(len(distanze)):
        error += distance_matrix[i, medoids_idx[labels[i]]]
    # Recupero i punti appartenenti ai cluster
    clusters = [[] for _ in range(t)]
    for i, label in enumerate(labels):
        clusters[label].append(distanze[i])

    return (key, {'medoids': medoids, 'clusters': clusters, 'error': error})


distributed_sampling_and_global_search(ds_import, 2, 120, 2)

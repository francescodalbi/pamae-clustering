from typing import Iterable, Tuple, NamedTuple, List
from collections.abc import Sequence

import random
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession
from sklearn.metrics.pairwise import manhattan_distances

# Create SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PAMAE") \
    .getOrCreate()

sc = spark.sparkContext
bin = 3
ds_import: ps.RDD[np.ndarray[float]] = sc.textFile("google_review_ratings.csv").map(lambda line: line.split(",")).map(
    lambda x: to_float_conversion(x)
)


def to_float_conversion(l: List[str]) -> np.ndarray[float]:
    float_lst = np.float_(l)
    return float_lst


print(type(ds_import.collect()[0]))
print(ds_import.collect()[0])

"""
def miafunz(x):
    print(x[0], x[1][0], x[1][0][0])
    print(" ")
    print(type(x[0]), type(x[1][0]), type(x[1][0][0]))
ds_samples.map(lambda x: miafunz(x)).collect()
"""


def distributed_sampling_and_global_search(
        dataset: ps.RDD[np.ndarray[float]],
        n_bins: int,
        sample_size: int,
        t: int
) -> np.ndarray:
    """
    Phase 1 of the algorithm presented in the PAMAE paper
    :param sample_size:
    :param n_bins:
    :param number_of_clusters:
    :param dataset:
    :return:
    """

    samples = get_random_samples(dataset, m=n_bins, n=sample_size)

    # TODO Implement distributed global search on each sample
    sample_medoids = []
    for i in range(0, t):
        best_medoids_i = global_search(samples, t)
        sample_medoids.append(best_medoids_i)

    # TODO find best medoids
    return min(sample_medoids)


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
    # TODO Genere chiave rnd, calcolo il modulo e lo assegno alle tuple

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
        ds_with_mod.groupByKey().mapValues(np.array).collect())

    def get_first_n_of_bin(bin_: Tuple[int, np.ndarray[np.ndarray[float]]]) -> Sample:
        return Sample(key=bin_[0], rows=bin_[1][:n])

    ds_samples: ps.RDD[Sample] = ds_grouped.map(get_first_n_of_bin)
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


def global_search(sample: np.ndarray | Iterable[float], k: int) -> SearchResult:
    """
    Phase I except for the sampling part

    :param sample: collection of objects (rows) from the dataset
    :param k: number of cluster (medoids)
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
    pass


def refinement(best_medoids: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Phase 2 of the algorithm presented in the PAMAE paper

    :param best_medoids: collection of the (ids of the) best medoids found in phase 1
    :param dataset: full dataset
    :return: array containing k clusters, each of which is represented by collection of data points
    """
    # 1. Identificare i cluster rispetto ai best_medoids e al dataset intero (punto 2. + 3. di global_search())
    # 2. Per ogni cluster_i. esegui global_search(sample=cluster_i, k=1)
    # 3. Con i medoidi ottenuti al punto 2., calcolo i cluster definitivi

    pass


def all_distances(sample: np.ndarray | Iterable[float]) -> np.ndarray:
    """
    >>> all_distances([[1, 2], [3, 4]])
    >>> np.ndarray([\
        [0., 4.],\
        [4., 0.]\
        ])

    :param sample: set of objects (dataset rows) sampled from the full dataset
    :return: 2D matrix where element ij is the distance between object (row) i and object (row) j
    """

    return manhattan_distances(sample, sample)

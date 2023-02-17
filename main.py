from typing import Iterable, Tuple, NamedTuple

import random
import numpy as np
from pyspark.sql import SparkSession
from sklearn.metrics.pairwise import manhattan_distances


# Create SparkSession
spark= SparkSession.builder \
      .master("local[*]") \
      .appName("PAMAE") \
      .getOrCreate()

sc = spark.sparkContext
bin = 3
ds_import= sc.textFile("google_review_ratings.csv").map(lambda line: line.split(",")).map(lambda x: conversione(x))

def conversione(x):
    #print(map(float, x))
    float_lst = list(np.float_(x))
    return float_lst

print(type(ds_import.collect()[0]))
print(ds_import.collect()[0])


### Genere chiave rnd, calcolo il modulo e lo assegno alle tuple
def random_mod(x):
    #m = numero di campioni desiderati
    rnd = random.randrange(0,9)
    mod = rnd % bin
    return mod, x

ds_with_mod = ds_import.map(lambda x: random_mod(x))


###

print(sorted(ds_with_mod.groupByKey().mapValues(len).collect()))

ds_grouped = sc.parallelize((ds_with_mod.groupByKey().mapValues(list).collect()))

#print(ds_grouped.take(1))

def myfunc(x):
    sample_size = 10

    #nuovo_array = np.empty([bin, 25], dtype=float)

    #nuovo_array = np.array(x[1][:sample_size])
    nuovo_array = np.array(x[1])
    print(nuovo_array.size)

    nuovo_array = np.insert(nuovo_array, 0, 99.99)
    #np.append(nuovo_array, added)

    return nuovo_array

#ds_samples = ds_grouped.map(lambda x: myfunc(x))
#print(ds_samples.collect())


n = 4 # numero di elementi che desideri prendere per ogni gruppo

ds_samples = ds_grouped.map(lambda x: (x[0], x[1][:n]))

#print(ds_samples.collect())


def miafunz(x):
    print(x[0], x[1][0], x[1][0][0])
    print(" ")
    print(type(x[0]), type(x[1][0]), type(x[1][0][0]))
ds_samples.map(lambda x: miafunz(x)).collect()


def phase_1(dataset: np.ndarray, k: int) -> np.ndarray:
    samples = get_random_samples(dataset)

    # TODO Distributed global search
    sample_medoids = []
    for i in range(0, k):
        best_medoids_i = global_search(samples, k)
        sample_medoids.append(best_medoids_i)

    # TODO find best medoids
    return min(sample_medoids)


class GlobalSearchResult(NamedTuple):
    medoid_ids: np.ndarray
    """
    Collection of ids that allow to identify the medoids in the full dataset
    """

    total_error: float
    """
    Sum of the distances 
    """


def global_search(sample: np.ndarray | Iterable[float], k: int) -> GlobalSearchResult:
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


def refinement(best_medoids: np.ndarray, dataset: np.ndarray) -> :
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




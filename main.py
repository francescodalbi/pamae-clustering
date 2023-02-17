from pyspark.sql import SparkSession
import sys
import random
import numpy as np

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
    m = 3
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


n = 2 # numero di elementi che desideri prendere per ogni gruppo



ds_samples = ds_grouped.map(lambda x: (x[0], x[1][:n]))

#print(ds_samples.collect()[1])




'''def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
'''

def miafunz(x):
    print(x[0], x[1])
ds_samples.map(lambda x: miafunz(x)).collect()



from sklearn.metrics.pairwise import pairwise_distances

from sklearn.metrics.pairwise import pairwise_distances
D = ds_samples.map(lambda x: (x[0], pairwise_distances(x[1], metric='euclidean')))
print(D.collect()[1])

#M = D.map(lambda x: (x[0], kMedoids(x[1], 2)))
#print(M.collect()[)


'''print('medoids:')
for point_idx in M.collect():
    print( data[point_idx] )

import kmedoids
km = kmedoids.KMedoids(2, method='fasterpam')
c = D.map(lambda x:km.fit(x[1]))
#print("Loss is:", c.take(1))
print(c.collect())
'''




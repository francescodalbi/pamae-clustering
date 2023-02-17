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

prova = sc.textFile("google_review_ratings.csv")
print(prova.collect()[2])
df = spark.read.csv("google_review_ratings.csv", header=True)
#df.printSchema()
#df.show()
### CONVERTO IL DATASET IN RDD

ds_rdd=df.rdd
#print(dati_rdd.collect())
#print(dati_rdd.count())
#print(dati_rdd.first())


### Genere chiave rnd, calcolo il modulo e lo assegno alle tuple

def random_mod(x):
    #m = numero di campioni desiderati
    m = 3
    rnd = random.randrange(0,9)
    mod = rnd % 3
    return mod, x

rdd_with_mod = ds_rdd.map(lambda x: random_mod(x))


###

print(sorted(rdd_with_mod.groupByKey().mapValues(len).collect()))

rdd_groupped = sc.parallelize((rdd_with_mod.groupByKey().mapValues(list).collect()))

#print(rdd_groupped.take(1))
#print(rdd_groupped.collect())

   #el1 = sc.parallelize(rdd_groupped[element])

def to_np(x):
    arr0 = np.empty(0)
    arr1 = np.empty(0)
    arr2 = np.empty(0)

    if x[0] == 0:
        aa = np.array(x[1])
        arr0 = np.append(arr0, aa)
    '''if x[0] == 1:
        aa = np.array(x[1])
        arr1 = np.append(arr1, aa)
    if x[0] == 2:
        aa = np.array(x[1])
        arr2 = np.append(arr2, aa)'''

    #prendo i primi 10 valori
    n=10

    #out_arr = np.add(arr0[:n],arr1[:n],arr2[:n])

    return arr0[0]


print(rdd_groupped.collect()[0])
#rdd_numpy = rdd_groupped.map(lambda x: to_np(x))
print(rdd_groupped.filter(lambda x: x[0] == 1).map(lambda x: x[1][1]).collect()[0])

#for element in rdd_groupped.collect():
    #print(element)


'''
arr0 = np.empty(0)
arr1 = np.empty(0)
arr2 = np.empty(0)

for element in rdd_groupped.collect():
    if element[0] == 0:
        aa = np.array(element[1])
        arr0 = np.append(arr0,aa)
    if element[0] == 1:
        aa = np.array(element[1])
        arr1 = np.append(arr1,aa)
    if element[0] == 2:
        aa = np.array(element[1])
        arr2 = np.append(arr2,aa)

print(arr0)
print(arr0.size)
print("###########à")
print(arr1)
'''





def kMedoids(D, k, tmax=100):
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

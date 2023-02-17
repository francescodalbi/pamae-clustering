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

#print(ds_samples.collect()[1])


def miafunz(x):
    print(x[0], x[1])
ds_samples.map(lambda x: miafunz(x)).collect()







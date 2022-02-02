#%%
import findspark
import pyspark
from pyspark.sql import SparkSession
# %%
findspark.init()
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('nuclear_plants_big_dataset.csv', inferSchema=True, header=True)
df.show()

#%%
"""Min & Max"""
my_rdd = df.rdd.map(lambda x: [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]])

# max:
max = my_rdd.reduce(lambda x, y: x if x > y else y)

# min:
min = my_rdd.reduce(lambda x, y: x if x < y else y)

#%%

my_second_rdd = df.rdd.map(lambda x: [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], 1])
mean_rdd = my_second_rdd.map(lambda x: (x[0], list(x[1:])))
mean = mean_rdd.reduceByKey(lambda x, y: (
    x[0] + y[0], 
    x[1] + y[1], 
    x[2] + y[2], 
    x[3] + y[3], 
    x[4] + y[4], 
    x[5] + y[5], 
    x[6] + y[6], 
    x[7] + y[7], 
    x[8] + y[8], 
    x[9] + y[9], 
    x[10] + y[10], 
    x[11] + y[11]    
))

data = mean.flatMap(lambda l: [(l[0], value) for value in l[1]]).toDF()
data_T = data.toPandas().T
mean_pd = data_T.drop(labels="_1")


#%%

cols = df.columns
cols = cols[1:]
# Right so we need to rename columns:
mean_pd.rename(columns = {
    0: cols[0],
    1: cols[1],
    2: cols[2],
    3: cols[3],
    4: cols[4],
    5: cols[5],
    6: cols[6],
    7: cols[7],
    8: cols[8],
    9: cols[9],
    10: cols[10],
    11: cols[11],
})

# %%

print(max)

# %%
# Format into a dictionary:
print(df.columns)
# %%
pdCols = {}
pdCols['Summary'] = ['Max']
for i, x in enumerate(max):
    pdCols[df.columns[i]] = [x]
    
print(pdCols)
# %%
import pandas as pd
eh = pd.DataFrame(data=pdCols)
# %%
eh
# %%
mean_pd.head()

# %%
import numpy as np
m = mean_pd.to_numpy()

print(m[0])

mean_arr = m[0]
mean_dc = {}
mean_dc['Summary'] = ['Mean']
mean_dc['Status'] = ['Normal']

cols = df.columns[1:]

for i, x in enumerate(mean_arr):
    mean_dc[cols[i]] = [x]
print(mean_dc)

# %%
ehh = pd.DataFrame(data=mean_dc)
ehh
# %%

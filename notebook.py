#%%
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd


# %%

findspark.init()
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True, header=True)
df.show()
# %%

Abnormal = df.where(df.Status == "Abnormal")
Abnormal.show()

# %%

d_abnormal = Abnormal.drop("Status")

# %%
"""This gets the row and count of each value in the dataset"""
my_c = d_abnormal.groupBy("Power_range_sensor_2").count()

# %%

"""Puts the dataframe into descending order"""
from pyspark.sql.functions import desc
c = my_c.orderBy('count', ascending=False)
# c.show()
# df.collect()[0]['count(DISTINCT AP)']
b = c.collect()[2]["Power_range_sensor_2"]
print(b)

#%%

abnomral_dict = {}

for c in d_abnormal.columns:
    temp = d_abnormal.groupBy(c).count()
    temp_2 = temp.orderBy('count', ascending=False)
    temp_2.show()

    print('mode: ', temp_2.collect()[0][c])
    abnomral_dict[c] = temp_2.collect()[0][c]

print(abnomral_dict)


#%%

Normal = df.where(df.Status == "Normal")
Normal.show()
d_normal = Normal.drop("Status")

normal_dict = {}

for c in d_normal.columns:
    temp = d_normal.groupBy(c).count().first[0]
    temp_2 = temp.orderBy('count', ascending=False)
    temp_2.show()

    print('mode: ', temp_2.collect()[0][c])
    normal_dict[c] = temp_2.collect()[0][c]
    

"""FINISH THIS !!!!!!!"""



#%%

"""Still Need to obtain the mode value!!!!!"""

"""Check this: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.first.html"""

mode = np.array(c.select('Power_range_sensor_2').orderBy('count', ascending=False).collect())
print(mode)

# %%

"""This works out the median"""

import pyspark.sql.functions as f 

# df_median = df.groupBy('Status').agg(f.percentile_approx())
df2 = df.groupBy('Status').agg([f.percentile_approx(c, 0.5).alias('Power_range_sensor_1_median') for c in df.columns])
df2.show()

# %%

"""Gets the values of the df as a numpy array"""

import numpy as np
a = np.array(df2.select('Power_range_sensor_1_median').collect()) # Abnormal is first
print(a)

# %%
"""Creates a pandas datafram to use to display things"""
import pandas as pd
d = { '---': ['Median', '---'],'Status': ['Abnormal', 'Normal'], 'Power_range_sensor_1_median': [a[0], a[1]]  }
b = pd.DataFrame(data=d)
b.head()
# %% 
n = a[0]
print(type(n))
print(float(n))
# %%

"""Correlation Matrix"""

def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)

Abnormal = df.where(df.Status == "Abnormal")
Normal = df.where(df.Status == "Normal")

Normal = Normal.drop("Status")
Abnormal = Abnormal.drop("Status")

corr_normal = correlation_matrix(df=Normal, corr_columns=Normal.columns)
corr_normal.head()

corr_abnormal = correlation_matrix(df=Abnormal, corr_columns=Abnormal.columns)
corr_abnormal.head()


# %%

"""Split training and test data"""

# Drop the status column for the ML data:
df_no_status = df.drop("Status")
df_train, df_test = df_no_status.randomSplit(weights=[70, 30], seed=200)
df_train.show()
# %%


# %%
N_Mean = Normal.select([_mean(c).alias(c) for c in Normal.columns])
N_Min = Normal.select([min(c).alias(c) for c in Normal.columns])
N_Max = Normal.select([max(c).alias(c) for c in Normal.columns])
N_Var = Normal.select([_var(c).alias(c) for c in Normal.columns])
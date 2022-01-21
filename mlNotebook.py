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

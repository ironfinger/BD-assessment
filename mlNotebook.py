#%%
from sre_parse import State
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
from sympy import im
# %%

findspark.init()
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True, header=True)
df.show()

# %%

# Change the status column to a 1 or 0
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="Status", outputCol="label")
indexed = indexer.fit(df).transform(df)

# Drop the status Column:
ml_df = indexed
ml_df.show()

# %%

# We need to get a list of the features:
cols = ml_df.columns
f = cols[:-1]
features = f[1:]
print(len(features))
# %%

vector_ass = VectorAssembler(inputCols=features, outputCol='features')
v_df = vector_ass.transform(ml_df)
v_df.show()

#%%

""" Split the Data """
splits = v_df.randomSplit([0.7, 0.3], 123)
train_df = splits[0]
test_df = splits[1]

#%%

"""Model Training and Validation"""

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

len(features)

layers = [len(features), 13, 13, 2]

# Create the neural network
mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)


# %%

"""Fit to the data"""
mlp_model = mlp.fit(train_df)
# %%

pred_df = mlp_model.transform(test_df)
pred_df.select('Power_range_sensor_1', 'features', 'label', 'rawPrediction', 'probability', 'prediction').show(5)

# %%

""" Evaluation """

evaluator = MulticlassClassificationEvaluator (labelCol='label', predictionCol='prediction', metricName='accuracy')
mlpacc = evaluator.evaluate(pred_df)
mlpacc
# %%
# https://cprosenjit.medium.com/9-classification-methods-from-spark-mllib-we-should-know-c41f555c0425

#%%

# Feature scaling:

from pyspark.ml.feature import StandardScaler

stdScaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=False)
scaler_model = stdScaler.fit(v_df)
scaled_df = scaler_model.transform(v_df)
scaled_df.select(['features', 'scaledFeatures']).show()

# %%
trainDF, testDF = scaled_df.randomSplit([.7, .3], seed=42)



#%%


from pyspark.ml.classification import LinearSVC
lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol='scaledFeatures', labelCol='label')

#%%
model = lsvc.fit(trainDF)

# %%

pred_df = model.transform(testDF)

# %%

# Accuracy:
lr_accuracy = evaluator.evaluate(pred_df)
print('Accuracy: ', lr_accuracy)
# %%

"""Desicion Tress"""

from pyspark.ml.classification import DecisionTreeClassifier

# Train a DescisionTree:
dt = DecisionTreeClassifier(labelCol='label', featuresCol='scaledFeatures', impurity='gini')

dt_train = dt.fit(trainDF)

#%%

""" Predict """
pred_dt = dt_train.transform(trainDF)

# %%

# Evaluation:
dt_accuracy = evaluator.evaluate(pred_dt)
print('accuracy: ', dt_accuracy)

# %%
pred_dt.show()

# %%

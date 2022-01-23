import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random

def neural_net(train_df, layers_m, test_df, seed=1):
    
    accuracies = []
    train_cyc = []
    train_i = 0
    
    for layers in layers_m:
        mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
        mlp_model = mlp.fit(train_df)
        pred_df = mlp_model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
        mlpaccc = evaluator.evaluate(pred_df)
        accuracies.append(mlpaccc)
        
        train_i += 1
        train_cyc.append(train_i)
        
        print('Training Cycle: ', len(train_cyc), 'Accuracy: ', mlpaccc, ' Neurons: ', layers)
        
    return train_cyc, accuracies
        
def main():
    
    # Import the data set:
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True, header=True)
    
    #  Index Status Label:
    indexer = StringIndexer(inputCol="Status", outputCol="label")
    ml_df = indexer.fit(df).transform(df)
    
    # Get features list:
    cols = ml_df.columns
    f = cols[:-1]
    features = f[1:]
    
    # Get features vector colulmn:
    vector_ass = VectorAssembler(inputCols=features, outputCol='features')
    ml_df = vector_ass.transform(ml_df)
    
    #Split the Data 
    splits = ml_df.randomSplit([0.7, 0.3], 123)
    train_df = splits[0]
    test_df = splits[1]

    layers_m = [
        [len(features), 15, 15 , 2],     
    ]
    
    epochs, accuracy = neural_net(train_df=train_df, layers_m=layers_m, test_df=test_df)
    
    plt.plot(epochs, accuracy)
    plt.show()
    
if "__main__" == __name__:
    main()
    
        
    
    
    
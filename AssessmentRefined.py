import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max, lit
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np

def get_sparkdf(spark, file_name):
    return spark.read.csv(file_name, inferSchema=True, header=True)

def missing_values_check(data):
    """
    Task 01
    """

    # Find IsNull:
    data.select([
        count(when(
            isnan(c) | 
            col(c).isNull(), 
            c)).alias(c) 
        for c in data.columns]).show()

    # Empty String literal:
    temp = data.select([
        count(when(
            col(c).contains('None') | \
            col(c).contains('NULL') | \
            (col(c) == '' ) | \
            col(c).isNull() | \
            isnan(c), c 
            )).alias(c)
            for c in data.columns])
    
    temp.show()

def show_summary(data, spark):

    # Get Normal and Abnormal datasets
    Normal = data.where(data.Status == "Normal")
    Normal = Normal.drop("Status")

    Abnormal = data.where(data.Status == "Abnormal")
    Abnormal = Abnormal.drop("Status")

    # Normal:
    N_Mean = Normal.select([_mean(c).alias(c) for c in Normal.columns])
    N_Min = Normal.select([min(c).alias(c) for c in Normal.columns])
    N_Max = Normal.select([max(c).alias(c) for c in Normal.columns])
    N_Var = Normal.select([_var(c).alias(c) for c in Normal.columns])

    mode_data = mode_dict(data=Normal)
    df = pd.DataFrame(mode_data)
    N_Mode = spark.createDataFrame(df)

    N_Mean = N_Mean.withColumn('Summary', lit('Mean'))
    N_Min = N_Min.withColumn('Summary', lit('Min'))
    N_Max = N_Max.withColumn('Summary', lit('Max'))
    N_Var = N_Var.withColumn('Summary', lit('Var'))
    N_Mode = N_Mode.withColumn('Summary', lit('Mode'))

    N_Mean = N_Mean.withColumn('Status', lit('Normal'))
    N_Min = N_Min.withColumn('Status', lit('Normal'))
    N_Max = N_Max.withColumn('Status', lit('Normal'))
    N_Var = N_Var.withColumn('Status', lit('Normal'))
    N_Mode = N_Mode.withColumn('Status', lit('Normal'))

    # Abnormal:
    A_Mean = Abnormal.select([_mean(c).alias(c) for c in Normal.columns])
    A_Min = Abnormal.select([min(c).alias(c) for c in Normal.columns])
    A_Max = Abnormal.select([max(c).alias(c) for c in Normal.columns])
    A_Var = Abnormal.select([_var(c).alias(c) for c in Normal.columns])

    mode_data = mode_dict(data=Abnormal)
    df = pd.DataFrame(mode_data)
    A_Mode = spark.createDataFrame(df)

    A_Mean = A_Mean.withColumn('Summary', lit('Mean'))
    A_Min = A_Min.withColumn('Summary', lit('Min'))
    A_Max = A_Max.withColumn('Summary', lit('Max'))
    A_Var = A_Var.withColumn('Summary', lit('Var'))
    A_Mode = A_Mode.withColumn('Summary', lit('Mode'))

    A_Mean = A_Mean.withColumn('Status', lit('Abnormal'))
    A_Min = A_Min.withColumn('Status', lit('Abnormal'))
    A_Max = A_Max.withColumn('Status', lit('Abnormal'))
    A_Var = A_Var.withColumn('Status', lit('Abnormal'))
    A_Mode = A_Mode.withColumn('Status', lit('Abnormal'))

    Union_df = N_Mean.union(N_Min)
    Union_df = Union_df.union(N_Max)
    Union_df = Union_df.union(N_Var)
    Union_df = Union_df.union(N_Mode)
    Union_df = Union_df.union(A_Mean)
    Union_df = Union_df.union(A_Min)
    Union_df = Union_df.union(A_Max)
    Union_df = Union_df.union(A_Var)
    Union_df = Union_df.union(A_Mode)
    Union_df = put_cols_to_left(Union_df, ['Status', 'Summary'])
    Union_df.show()
    
def put_cols_to_left(data, cols_to_left):
    original_cols = data.columns
    ordered_cols = cols_to_left

    for i, c in enumerate(original_cols):
        if c != ordered_cols[i]:
            ordered_cols.append(c)

    print(ordered_cols)
    return data.select(ordered_cols[:-2])

def mode_dict(data):
    mode_dict = {}

    for c in data.columns:
        temp = data.groupBy(c).count()
        temp_2 = temp.orderBy('count', ascending=False)
        
        mode_dict[c] = [temp_2.collect()[0][c]]

    return mode_dict

def display_boxplots(data):
    df_pd = data.toPandas() # Convert the spark df to pandas.
    df_pd.boxplot(by="Status")
    plt.show()

def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=corr_columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)
    
def index_label(df, colInput, colOutput='label'):
    indexer = StringIndexer(inputCol=colInput, outputCol=colOutput)
    return indexer.fit(df).transform(df)

def get_features(df, features, colOutput='features'):
    vector_assemble = VectorAssembler(inputCols=features, outputCol=colOutput)
    return vector_assemble.transform(df)
    
def get_train_test(df, seed, split=[0.7, 0.3]):
    splits = df.randomSplit(split, seed)
    train_df = splits[0]
    test_df = splits[1]
    return train_df, test_df

def get_scaled_features(df, inputCol='features', outputCol='scaledFeatures'):
    std_scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=False)
    scaler_model = std_scaler.fit(df)
    return scaler_model.transform(df)

def generate_train_test(df):
    # Get the features:
    cols = df.columns
    f = cols[:-1]
    features = f[1:]
    
    # Get label:
    df = index_label(df=df, colInput='Status')
    
    # Get features:
    df = get_features(df=df, features=features)
    
    # Scale Features
    df = get_scaled_features(df=df)
    
    train_df, test_df = get_train_test(df=df, seed=123)
    
    return train_df, test_df

def desision_tree(df):

    # Get train and test data:
    train_df, test_df = generate_train_test(df=df)
    
    # Make Desicion Tree Obj:
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='scaledFeatures', impurity='gini')
    
    # Train the model:
    dt_train = dt.fit(train_df)
    
    # Predict:
    dt_pred = dt_train.transform(test_df)
    
    # Evaluation:
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label', 
        predictionCol='prediction', 
        metricName='accuracy'
    )
    
    mlpacc = evaluator.evaluate(dt_pred)
    return dt_pred
    
def support_vector_machine(df):
    
    # Get the train and test data:
    train_df, test_df = generate_train_test(df=df)
    
    # Make the support vector machine:
    lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol='scaledFeatures', labelCol='label')
    
    # Fit the model:
    model = lsvc.fit(train_df)
    
    # Make the predictions:
    pred_df = model.transform(test_df)
    
    # Evaluation:
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label', 
        predictionCol='prediction', 
        metricName='accuracy'
    )
    
    mlpacc = evaluator.evaluate(pred_df)
    return pred_df

def neural_net(df):
    
    indexer = StringIndexer(inputCol="Status", outputCol="label")
    ml_df = indexer.fit(df).transform(df)
    
    cols = ml_df.columns
    f = cols[:-1]
    features = f[1:]
    
    # Get features vector column:
    vector_ass = VectorAssembler(inputCols=features, outputCol='features')
    ml_df = vector_ass.transform(ml_df)
    
    # Split the data:
    splits = ml_df.randomSplit([0.7, 0.3], 123)
    train_df = splits[0]
    test_df = splits[1]
    
    layers = [len(features), 15, 15, 2]
    
    mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
    mlp_model = mlp.fit(train_df)
    pred_df = mlp_model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    mlpacc = evaluator.evaluate(pred_df)
    
    return pred_df

def error_rate(pred_df):
  
    normal = pred_df.where(pred_df.Status == "Normal")
    abnormal = pred_df.where(pred_df.Status == "Abnormal")
    
    N_incorrect = normal.where(normal.prediction == 0)
    A_incorrect = abnormal.where(abnormal.prediction == 1)
    
    n = N_incorrect.count()
    a = A_incorrect.count()
    p = pred_df.count()
    
    error = (n + a) / p
    return error

def get_confusion_m(pred_df):
    # from pyspark.mllib.evaluation import MulticlassMetrics
    # pred_labels = nn_pred.select(['label', 'prediction'])
    # pred_labels.show()
    
    # metrics = MulticlassMetrics(pred_labels.rdd.map(lambda x: tuple(map(float, x))))
    # confusion = metrics.confusionMatrix().toArray()
    # labels = [int(l) for l in metrics.call('labels')]
    # confusion_m = pd.DataFrame(confusion, index=labels, columns=labels)
    # print(confusion_m)
    
    pred_labels = pred_df.select(['label', 'prediction'])
    
    metrics = MulticlassMetrics(pred_labels.rdd.map(lambda x: tuple(map(float, x))))
    confusion = metrics.confusionMatrix().toArray()
    labels = [int(l) for l in metrics.call('labels')]
    confusion_m = pd.DataFrame(confusion, index=labels, columns=labels)
    return confusion_m

def confusion_v2(pred_df):
    total_test_rows = pred_df.count()
    print('total test rows: ', total_test_rows)
    
    # True Negative (0, 0) = Abnormal Abnormal
    # False Positive (0, 1) = Abnormal (Normal)
    # False Negative (1, 0) = Normal (Abnormal)
    # True Positive (1, 1) = Normal Normal
    
    # Abnormal -> Abnormal || 0 -> 0
    true_negatve = pred_df.where((col('prediction')=='0') & (col('label')=='0')).count()
    
    # Abnormal -> Normal || 0 -> 1
    false_positive = pred_df.where((col('prediction')=='1') & (col('label')=='0')).count()
    
    # Normal -> Normal || 1 -> 1
    true_positive = pred_df.where((col('prediction')=='1') & (col('label')=='1')).count()
    
    # Normal -> Abnormal || 1 -> 0
    false_negative = pred_df.where((col('prediction')=='0') & (col('label')=='1')).count()
    
    confusion_m = np.array([
        [true_negatve, false_positive],
        [false_negative, true_positive]
    ])
    
    print('confusion matrix')
    print(confusion_m)
    
    return confusion_m
    
def get_specifity_sensitivity(pred_df):
    confusion_m = get_confusion_m(pred_df=pred_df)
    
    print('Confusion Matrix: ')
    print(confusion_m)
    
    confusion_m_np = confusion_m.to_numpy()
    
    print('Confusion matrix')
    print(confusion_m)
    
    true_positive = confusion_m_np[0][0]
    false_negative = confusion_m_np[1][0]
    
    true_negative = confusion_m_np[1][1]
    false_positive = confusion_m_np[0][1]
    
    sensitivity = true_positive / (true_positive + false_negative)
    specifity = true_negative / (true_negative + false_positive)
    
    return sensitivity, specifity
    
def get_sens_speci(pred_df):
    confusion_m = confusion_v2(pred_df=pred_df)
    
    # Sensitivity = true positive / (tru pos, fal neg)
    # Sensitivity = true Normal / (true normals + false normals)
    
    # Specificity = true abnormal / (true abnormal + false abnormals)
    true_negative = confusion_m[0][0]
    false_positive = confusion_m[0][1]
    true_positive = confusion_m[1][1]
    false_negative = confusion_m[1][0]
    
    sensitivity_brackets = true_positive + false_negative
    specificity_brackets = true_negative + false_positive
    
    sensitivity = true_positive / sensitivity_brackets
    specificity = true_negative / specificity_brackets
    
    return sensitivity, specificity
    

def main():

    # Abnormal = 0
    # Normal = 1
    
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    df = get_sparkdf(spark=spark, file_name="nuclear_plants_small_dataset.csv")
    
    # print('MISSING VALUES')
    # missing_values_check(data=df)

    print('SUMMARY')
    show_summary(data=df, spark=spark)
    
    Abnormal = df.where(df.Status == "Abnormal")
    Normal = df.where(df.Status == "Normal")

    Normal = Normal.drop("Status")
    Abnormal = Abnormal.drop("Status")

    # THIS CODE MAXIMUS!!!!!
    corr_normal = correlation_matrix(df=Normal, corr_columns=Normal.columns)
    print('correlation matrix normal')
    print(corr_normal)

    corr_abnormal = correlation_matrix(df=Abnormal, corr_columns=Abnormal.columns)
    print('correlation matrix abnormal')
    print(corr_abnormal)
    
    """Machine learning"""
    print('Descision Tree Accuracy: ', desision_tree(df=df))
    print('Support Vector Machine Accuracy: ', support_vector_machine(df=df))
    
    nn_pred = neural_net(df=df)
    dt_pred = desision_tree(df=df)
    svm_pred = support_vector_machine(df=df)
    
    print('Values I need')
    print('Normal')
    pred_normal = nn_pred.where(nn_pred.Status == 'Normal')
    pred_abnormal = nn_pred.where(nn_pred.Status == 'Abnormal')
    
    values_i_need = pred_normal.select(['prediction', 'label', 'Status'])
    values_i_need.show()
    
    values_i_need = pred_abnormal.select(['prediction', 'label', 'Status'])
    values_i_need.show()
    
    nn_error = error_rate(pred_df=nn_pred)
    dt_error = error_rate(pred_df=dt_pred)
    svm_error = error_rate(pred_df=svm_pred)
    
    print('Neural Net Error: ', nn_error)
    print('Desicion Tree Error: ', dt_error)
    print('Support Vector Error: ', svm_error)
    
    nn_sens, nn_speci = get_sens_speci(pred_df=nn_pred)
    dt_sens, dt_speci = get_sens_speci(pred_df=dt_pred)
    svm_sens, svm_speci = get_sens_speci(pred_df=svm_pred)
    
    print('--- NEURAL NETWORK ---')
    print('Sensitivity: ', nn_sens, ' Specitivity: ', nn_speci)
    print('--- Descision Tree ---')
    print('Sensitivity: ', dt_sens, ' Specitivity: ', dt_speci)
    print('--- Support Vector Machine ---')
    print('Sensitivity: ', svm_sens, ' Specitivity: ', svm_speci)
    


if __name__ == "__main__":
    main()
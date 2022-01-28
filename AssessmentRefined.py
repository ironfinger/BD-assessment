from webbrowser import get
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
    
def get_specifity_sensitivity(pred_df):
    confusion_m = get_confusion_m(pred_df=pred_df)
    confusion_m_np = confusion_m.to_numpy()
    
    true_positive = confusion_m_np[0][0]
    false_negative = confusion_m_np[1][0]
    
    true_negative = confusion_m_np[1][1]
    false_positive = confusion_m_np[0][1]
    
    sensitivity = true_positive / (true_positive + false_negative)
    specifity = true_negative / (true_negative + false_positive)
    
    return sensitivity, specifity
    

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

    nn_error = error_rate(pred_df=nn_pred)
    dt_error = error_rate(pred_df=dt_pred)
    svm_error = error_rate(pred_df=svm_pred)
    
    print('Neural Net Error: ', nn_error)
    print('Desicion Tree Error: ', dt_error)
    print('Support Vector Error: ', svm_error)
    
    nn_sens, nn_speci = get_specifity_sensitivity(pred_df=nn_pred)
    dt_sens, dt_speci = get_specifity_sensitivity(pred_df=dt_pred)
    svm_sens, svm_speci = get_specifity_sensitivity(pred_df=svm_pred)
    
    print('--- NEURAL NETWORK ---')
    print('Sensitivity: ', nn_sens, ' Specitivity: ', nn_speci)
    print('--- Descision Tree ---')
    print('Sensitivity: ', dt_sens, ' Specitivity: ', dt_speci)
    print('--- Support Vector Machine ---')
    print('Sensitivity: ', svm_sens, ' Specitivity: ', svm_speci)
    
    # Obtain the confusion matrix's for the following models/predictions:
    # nn_confusion = get_confusion_m(nn_pred)
    # dt_confusion = get_confusion_m(dt_pred)
    # svm_confusion = get_confusion_m(svm_pred)
    
    # print('Neural net confusion')
    # print(nn_confusion)
    
    # print('')
    # print('Neural Net confusion as np array')
    # print(nn_confusion.to_numpy())
    # nn_np_confusion = nn_confusion.to_numpy()
    # Sensitivity = true positive / (true positive + false negative)
    # Specifity = true negatives / (true negative + false positive)
    
    # Sensitivity = 113 / (133 + 36)
    
    # True positive = [0][0]
    # false negative = [1][0]
    
    # True negative: [1][1]
    # False positive: [0][1]
    
    # true_positive = nn_np_confusion[0][0]
    # false_negative = nn_np_confusion[1][0]
    
    # true_negative = nn_np_confusion[1][1]
    # false_positive = nn_np_confusion[0][1]
    
    # sensitivity = true_positive / (true_positive + false_negative)
    # specifity = true_negative / (true_negative + false_positive)
    
    # print('sensitivity: ', sensitivity, ' | specifity: ', specifity)

if __name__ == "__main__":
    main()
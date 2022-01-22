from ctypes import Union
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max, lit
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
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

def show_mean(data, spark):

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



"""

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
"""

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
    


    
def main():

    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    df = get_sparkdf(spark=spark, file_name="nuclear_plants_small_dataset.csv")
    
    # print('MISSING VALUES')
    # missing_values_check(data=df)

    print('SUMMARY')
    show_mean(data=df, spark=spark)
    
    Abnormal = df.where(df.Status == "Abnormal")
    Normal = df.where(df.Status == "Normal")

    Normal = Normal.drop("Status")
    Abnormal = Abnormal.drop("Status")

    # THIS CODE MAXIMUS!!!!!
    corr_normal = correlation_matrix(df=Normal, corr_columns=Normal.columns)
    print(corr_normal)

    # corr_abnormal = correlation_matrix(df=Abnormal, corr_columns=Abnormal.columns)
    # print(corr_abnormal)

    # print('DONE')



if __name__ == "__main__":
    main()
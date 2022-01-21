import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max
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



def show_mean(data):

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

    # Mode:
    N_mode = mode_dict(data=Normal)
    panda_bear = pd.DataFrame(data=N_mode)
    print(panda_bear)

    # Abnormal:
    A_Mean = Abnormal.select([_mean(c).alias(c) for c in Normal.columns])
    A_Min = Abnormal.select([min(c).alias(c) for c in Normal.columns])
    A_Max = Abnormal.select([max(c).alias(c) for c in Normal.columns])
    A_Var = Abnormal.select([_var(c).alias(c) for c in Normal.columns])

    # Mode
    A_mode = mode_dict(data=Abnormal)
    panda_bear_2 = pd.DataFrame(data=A_mode)
    print(panda_bear_2)

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
    
    print('MISSING VALUES')
    missing_values_check(data=df)

    print('SUMMARY')
    show_mean(data=df)
    
    Abnormal = df.where(df.Status == "Abnormal")
    Normal = df.where(df.Status == "Normal")

    Normal = Normal.drop("Status")
    Abnormal = Abnormal.drop("Status")

    corr_normal = correlation_matrix(df=Normal, corr_columns=Normal.columns)
    print(corr_normal)

    corr_abnormal = correlation_matrix(df=Abnormal, corr_columns=Abnormal.columns)
    print(corr_abnormal)

    print('DONE')



if __name__ == "__main__":
    main()
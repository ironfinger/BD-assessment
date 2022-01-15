import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, variance as _var, col, isnan, when, count, min, max
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd



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



# def show_summary(data):
    
#     Normal = data.where(data.Status == "Normal")
#     Abnormal = data.where(data.Status == "Abnormal")

#     # Drop the status column:
#     Normal = Normal.drop("Status")
#     Abnormal = Abnormal.drop("Status")

#     # Normal Statistics:
#     Mean_Normal = Normal.select(*[_mean(c).alias(c) for c in Normal.columns])
#     Min_Normal = Normal.select(*[min(col(c)).alias(c) for c in Normal.columns])
#     Max_Normal = Normal.select(*[max(col(c)).alias(c) for c in Normal.columns])
#     Variance_Normal = Normal.select(*[_var(col(c)).alias(c) for c in Normal.columns])

#     # Abnormal Statistics:
#     Mean_Abnormal = Abnormal.select(*[_mean(c).alias(c) for c in Abnormal.columns])
#     Min_Abnormal = Abnormal.select(*[min(col(c)).alias(c) for c in Abnormal.columns])
#     Max_Abnormal = Abnormal.select(*[max(col(c)).alias(c) for c in Abnormal.columns])
#     Variance_Abnormal = Abnormal.select(*[_var(col(c)).alias(c) for c in Abnormal.columns])

#     print('---Normal---')
#     print('--Mean--')
#     Mean_Normal.show()
#     print('--Min--')
#     Min_Normal.show()
#     print('--Max--')
#     Max_Normal.show()
#     print('--Variance--')
#     Variance_Normal.show()

#     print('---Abnormal---')
#     Mean_Abnormal.show()
#     Min_Abnormal.show()
#     Max_Abnormal.show()
#     Variance_Normal.show()

def show_mean(data):
    Normal = data.where(data.Status == "Normal")
    Normal = Normal.drop("Status")

    Abnormal = data.where(data.Status == "Abnormal")
    Abnormal = Abnormal.drop("Status")

    # Normal:
    N_Mean = Normal.select([_mean(c).alias(c) for c in Normal.columns])
    N_Min = Normal.select([min(c).alias(c) for c in Normal.columns])
    N_Max = Normal.select([max(c).alias(c) for c in Normal.columns])
    N_Var = Normal.select([_var(c).alias(c) for c in Normal.columns])

    # Abnormal:
    A_Mean = Abnormal.select([_mean(c).alias(c) for c in Normal.columns])
    A_Min = Abnormal.select([min(c).alias(c) for c in Normal.columns])
    A_Max = Abnormal.select([max(c).alias(c) for c in Normal.columns])
    A_Var = Abnormal.select([_var(c).alias(c) for c in Normal.columns])

    print('Normal Mean')
    N_Var.show()

    print('Abnormal Mean')
    A_Mean.show()
   
    

def display_boxplots(data):
    df_pd = data.toPandas() # Convert the spark df to pandas.
    df_pd.boxplot(by="Status")
    plt.show()


def corr_matrix(data):

    data_rdd = data.rdd.map(lambda row: row[0:])
    corr_mat = Correlation.corr(data_rdd, method='pearson')
    corr_mat_pd = pd.DataFrame(
        corr_mat,
        columns=data.columns,
        index=data.columns
    )

    return corr_mat_pd


    
def main():

    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    df = get_sparkdf(spark=spark, file_name="nuclear_plants_small_dataset.csv")
    
    print('MISSING VALUES')
    missing_values_check(data=df)

    print('SUMMARY')
    show_mean(data=df)
    #display_boxplots(data=df)
    print(corr_matrix(data=df))
    print('DONE')



if __name__ == "__main__":
    main()
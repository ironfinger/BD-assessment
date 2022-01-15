#%%
import findspark
import pyspark
from pyspark.sql import SparkSession

findspark.init()

spark = SparkSession.builder.getOrCreate()

# Load the data into a pyspark data frame:
df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
df.show() # Show the data.

# %%

"""
Task 01: Before making any analysis, it is required to know if there are missing values in the data.

Are there any missing values ?
How will you deal with the missing values.

Why would we have missing data: https://www.analyticsvidhya.com/blog/2021/06/the-missing-data-understand-the-concept-behind/

Types of missing data:

- Missing completely at random MCAR -> Data is just randomly missing.
- Missing at random MAR -> Missing conditionally at random based on another observation.
- Missing not at random MNAR -> Missing as part of how it is collected.

- We need to count the rows that are missing.

We can determine if there are missing fields using the spark sql functions "isnan" and the spark column function "isNull".
Link -> https://sparkbyexamples.com/pyspark/pyspark-find-count-of-null-none-nan-values/

"""

# A list of all the columns in the data frame:
df_column_names = [
    "Power_range_sensor_1", 
    "Power_range_sensor_2", 
    "Power_range_sensor_3", 
    "Power_range_sensor_4", 
    "Pressure _sensor_1", 
    "Pressure _sensor_2", 
    "Pressure _sensor_3", 
    "Pressure _sensor_4", 
    "Vibration_sensor_1", 
    "Vibration_sensor_2", 
    "Vibration_sensor_3", 
    "Vibration_sensor4"
]

df_column_locations = [
    "_c1",
    "_c2",
    "_c3",
    "_c4",
    "_c5",
    "_c6",
    "_c7",
    "_c8",
    "_c9",
    "_c10",
    "_c11",
    "_c12"
]

from pyspark.sql.functions import col, isnan, when, count

# This function checks to see if there are any Null or Nan values:
def find_isNull_columns(data):
    data.select([
        count(when(
            isnan(c) | 
            col(c).isNull(), 
            c)).alias(c) 
        for c in data.columns]).show()

# This function checks to see if there are any missing values in the form of a string:
def find_empty_string_literal_values(data):
    temp = df.select([
        count(when(
            col(c).contains('None') | \
            col(c).contains('NULL') | \
            (col(c) == '' ) | \
            col(c).isNull() | \
            isnan(c), c 
            )).alias(c)
            for c in df.columns])
    
    return temp


# Call these functions to display the count of these missing values in each column:
print('--- Check to See if there are any missing values ---')
find_isNull_columns(df)
df2 = find_empty_string_literal_values(df)
df2.show()

# %%
"""
Task 2: Before making an analysis, it is benificial to understand the data by looking at the summary statistics.
There are two groups of subjects ( the normal group and the abnormal group)
For each group show the folowing summary statisics for each feature in a table:
Minimum, maximum, mean, meadian, mode and variance values.
For each group plot the box plot for each feature.
"""

# Create a new table for just the normal group:

# Create a new table for just the abnormal group:


import pandas as pd
import numpy as np
# Modifying the print parameters so that we can print information about all the columns of our dataframes
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Loading the train and test datasets
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test-full.csv")

# Counting the number of lines of the train and test sets
print("Train set has "+str(len(train_df.index))+" lines")
print("Test set has "+str(len(test_df.index))+" lines")

# Displaying basic statistics about the train and the test set
print("Statistics of train set")
print(train_df.describe())
print("Statistics of test set")
print(test_df.describe())

# Displaying the number of unique values in the train and test set
print("Number of unique values in train set")
for col in train_df.columns:
    print(col+": "+str(len(train_df[col].unique())))
print("Number of unique values in test set")
for col in test_df.columns:
    print(col+": "+str(len(test_df[col].unique())))

# Check if there are non-numeric values in the train and the test set
print("Checking for nans")
print("Nans in the train set: "+str(train_df.isnull().values.any()))
print("Nans in the test set: "+str(test_df.isnull().values.any()))

# Check if there are defaults in the one-hot encoding of the soil type in the train and the test set
print("Checking for multiple soil types in the train set")
soiltypes = train_df[["Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40"]]
soiltypes["sum"] = soiltypes.sum(axis=1)
print("The following number should be 1: "+str(soiltypes["sum"].unique()))
print("Checking for multiple soil types in the test set")
soiltypes = test_df[["Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40"]]
soiltypes["sum"] = soiltypes.sum(axis=1)
print("The following number should be 1: "+str(soiltypes["sum"].unique()))

# The following function counts the number of large outlier for the column col of the dataframe df
# We define as large outliers any outlier further than 4 interquartile ranges away from the mean of the column
def count_large_outliers(df, col):
    # Computing the interquartile range
    Q1 = np.percentile(np.array(df[col].values),25)
    Q3 = np.percentile(np.array(df[col].values),75)
    IQ_range = Q3 - Q1
    # Computing the bounds of the normal values interval             
    high_bound = Q3 + 5 * IQ_range
    low_bound = Q1 - 5 * IQ_range
    # Counting large outliers
    count = 0                 
    for x in df[col].values:
        if (x < low_bound) or (x > high_bound):
            count += 1
    return count

# Printing the number of outliers for each column of the train set
'''for col in train_df.columns:
    if (count_large_outliers(train_df, col) > 0) and (!(col.startswith("Wilderness_Area"))) and (!(col.startswith("Soil_Type"))):
        print("There are "+str(count_large_outliers(train_df,col))+" outliers in "+str(col))'''
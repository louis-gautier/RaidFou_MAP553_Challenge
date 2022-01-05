# Feature design: new features for soil type
import pandas as pd
import numpy as np

# Load the train and the test set
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test-full.csv")

# This function adds new features for soil type: reverse one-hot encoding: climatic and geologic zone as a function of the ELU code
def add_soil_type_info(df):
  soil_type_climatic_zone = []
  soil_type_geologic_zone = []
  for index, row in df.iterrows():
    if row['Soil_Type1'] + row['Soil_Type2'] + row['Soil_Type3'] + row['Soil_Type4'] + row['Soil_Type5'] + row['Soil_Type6'] > 0:
      soil_type_climatic_zone.append("ClimaticGlacial")
      soil_type_geologic_zone.append("GeologicIgneous")
      continue
    if row['Soil_Type7'] + row['Soil_Type8'] > 0:
      soil_type_climatic_zone.append("ClimaticMontaneDry")
      soil_type_geologic_zone.append("GeologicMixedSedimentary")
      continue
    if row['Soil_Type9'] + row['Soil_Type10'] + row['Soil_Type11'] + row['Soil_Type12'] + row['Soil_Type13'] > 0:
      soil_type_climatic_zone.append("ClimaticMontane")
      if(row['Soil_Type9']==1):
        soil_type_geologic_zone.append("GeologicGlacial")
      else:
        soil_type_geologic_zone.append("GeologicIgneous")
      continue
    if row['Soil_Type14'] + row['Soil_Type15'] > 0:
      soil_type_climatic_zone.append("ClimaticMontaneDryMontane")
      soil_type_geologic_zone.append("GeologicAlluvium")
      continue
    if row['Soil_Type16'] + row['Soil_Type17'] + row['Soil_Type18'] > 0:
      soil_type_climatic_zone.append("ClimaticMontaneSubalpine")
      if(row['Soil_Type18']==1):
        soil_type_geologic_zone.append("GeologicIgneous")
      else:
        soil_type_geologic_zone.append("GeologicAlluvium")
      continue
    if row['Soil_Type19'] + row['Soil_Type20'] + row['Soil_Type21'] + row['Soil_Type22'] + row['Soil_Type23'] + row['Soil_Type24'] + row['Soil_Type25'] + row['Soil_Type26'] + row['Soil_Type27'] + row['Soil_Type28'] + row['Soil_Type29'] + + row['Soil_Type30'] + row['Soil_Type31'] + row['Soil_Type32'] + row['Soil_Type33'] + row['Soil_Type34'] > 0:
      soil_type_climatic_zone.append("ClimaticSubalpine")
      if(row['Soil_Type19'] + row['Soil_Type20'] + row['Soil_Type21'] > 0):
        soil_type_geologic_zone.append("GeologicAlluvium")
      elif(row['Soil_Type22'] + row['Soil_Type23'] > 0):
        soil_type_geologic_zone.append("GeologicGlacial")
      else:
        soil_type_geologic_zone.append("GeologicIgneous")
      continue
    if row['Soil_Type35'] + row['Soil_Type36'] + row['Soil_Type37'] + row['Soil_Type38'] + row['Soil_Type39'] + row['Soil_Type40'] > 0:
      soil_type_climatic_zone.append("ClimaticAlpine")
      soil_type_geologic_zone.append("GeologicIgneous")
      continue
  clim_zone = pd.get_dummies(soil_type_climatic_zone)
  geologic_zone = pd.get_dummies(soil_type_geologic_zone)
  concat = pd.concat([clim_zone, geologic_zone],axis=1)
  concat["Id"] = df.index
  concat.set_index("Id",inplace=True)
  return concat

# We add the new features to the train set
train_df = pd.concat([train_df,add_soil_type_info(train_df)],axis=1)
train_df.set_index("Id",inplace=True)
# We reverse one hot encoding of the soil type
filter_soiltype = [col for col in train_df if col.startswith("Soil_Type")]
soiltypes_train = train_df[filter_soiltype]
soiltypes_indices = (soiltypes_train.iloc[:] == 1).idxmax(1)
soiltypes_indices = [soiltypes_train.columns.get_loc(x)+1 for x in soiltypes_indices.values]
train_df.drop(filter_soiltype,axis=1,inplace=True)
train_df["Soil_Type"]=soiltypes_indices

# We add the new features to the test set
test_df = pd.concat([test_df,add_soil_type_info(test_df)],axis=1)
test_df.set_index("Id",inplace=True)
# We reverse one hot encoding of the soil type
filter_soiltype = [col for col in test_df if col.startswith("Soil_Type")]
soiltypes_test = test_df[filter_soiltype]
soiltypes_indices = (soiltypes_test.iloc[:] == 1).idxmax(1)
soiltypes_indices = [soiltypes_test.columns.get_loc(x)+1 for x in soiltypes_indices.values]
test_df.drop(filter_soiltype,axis=1,inplace=True)
test_df["Soil_Type"]=soiltypes_indices

# Writing train and test set to disk
<<<<<<< HEAD
train_df.to_csv("../data/train-temp.csv",index=True)
test_df.to_csv("../data/test-temp.csv",index=True)
=======
train_df.to_csv("drive/MyDrive/data553/train-temp.csv",index=True)
test_df.to_csv("drive/MyDrive/data553/test-temp.csv",index=True)
>>>>>>> 87c858b435175c6a1b9ce3ecf7cbf43e66305bd0

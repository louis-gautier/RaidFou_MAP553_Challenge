import pandas as pd
import numpy as np

SCENARIO = "plusLogTransformation"


# Load the train and the test set with extra soil type features
train_df = pd.read_csv("../data/train-temp.csv",index_col=0)
test_df = pd.read_csv("../data/test-temp.csv",index_col=0)

#train_df = train_df[(train_df['Horizontal_Distance_To_Fire_Points'] > outlier_function(train_df, 'Horizontal_Distance_To_Fire_Points')[0]) &
#              (train_df['Horizontal_Distance_To_Fire_Points'] < outlier_function(train_df, 'Horizontal_Distance_To_Fire_Points')[1])]

# Compute the euclidean distance to hydrology
train_df['Distance_To_Hydrology'] = np.sqrt(train_df['Horizontal_Distance_To_Hydrology']**2 + train_df['Vertical_Distance_To_Hydrology']**2)
# Compute linear combinations of horizontal distances
train_df['Hydrology_Plus_Firepoints'] = train_df['Horizontal_Distance_To_Hydrology'] + train_df['Horizontal_Distance_To_Fire_Points']
train_df['Hydrology_Minus_Firepoints'] = train_df['Horizontal_Distance_To_Hydrology'] - train_df['Horizontal_Distance_To_Fire_Points']
train_df['Hydrology_Plus_Roadways'] = train_df['Horizontal_Distance_To_Hydrology'] + train_df['Horizontal_Distance_To_Roadways']
train_df['Hydrology_Minus_Roadways'] = train_df['Horizontal_Distance_To_Hydrology'] - train_df['Horizontal_Distance_To_Roadways']
train_df['Roadways_Plus_Firepoints'] = train_df['Horizontal_Distance_To_Roadways'] + train_df['Horizontal_Distance_To_Fire_Points']
train_df['Roadways_Minus_Firepoints'] = train_df['Horizontal_Distance_To_Roadways'] - train_df['Horizontal_Distance_To_Fire_Points']
# Compute linear combinations of vertical distances
train_df['Elevation_Plus_Vertical_Distance_Hydrology'] = train_df['Elevation'] + train_df['Vertical_Distance_To_Hydrology']
train_df['Elevation_Minus_Vertical_Distance_Hydrology'] = train_df['Elevation'] - train_df['Vertical_Distance_To_Hydrology']
# Compute extra features for Aspect
train_df['cos_plus_sinAspect'] = np.cos((train_df['Aspect']*np.pi)/180) + np.sin((train_df['Aspect']*np.pi)/180)
# Compute linear combinations of hillshade variables
train_df['Hillshade-9_Minus_Noon'] = train_df['Hillshade_9am'] - train_df['Hillshade_Noon']
train_df['Hillshade-noon_Minus_3pm'] = train_df['Hillshade_Noon'] - train_df['Hillshade_3pm']
train_df["Hillshade-9am_Minus_3pm"] = train_df["Hillshade_9am"] - train_df["Hillshade_3pm"]

# Compute log transformations of positive features
for col in train_df.columns:
    if train_df[col].min() >= 0:
        if col == 'Cover_Type' or col.startswith('Geologic') or col.startswith('Climatic') or col.startswith('Wilderness'):
            next
        else:
            train_df['log' + col] = np.log(1+train_df[col])

# Print features sorted by Pearson correlation coefficient with cover type
corr = pd.DataFrame(train_df.corr())
corr = pd.DataFrame(corr["Cover_Type"]).reset_index()
corr.columns = ["Feature", "Correlation"]
corr = (corr[corr["Feature"] != "Cover_Type"].sort_values(by="Correlation"))
print(corr)

# Compute the euclidean distance to hydrology on the test set
test_df['Distance_To_Hydrology'] = np.sqrt(test_df['Horizontal_Distance_To_Hydrology']**2 + test_df['Vertical_Distance_To_Hydrology']**2)
# Compute linear combinations of horizontal distances on the test set
test_df['Hydrology_Plus_Firepoints'] = test_df['Horizontal_Distance_To_Hydrology'] + test_df['Horizontal_Distance_To_Fire_Points']
test_df['Hydrology_Minus_Firepoints'] = test_df['Horizontal_Distance_To_Hydrology'] - test_df['Horizontal_Distance_To_Fire_Points']
test_df['Hydrology_Plus_Roadways'] = test_df['Horizontal_Distance_To_Hydrology'] + test_df['Horizontal_Distance_To_Roadways']
test_df['Hydrology_Minus_Roadways'] = test_df['Horizontal_Distance_To_Hydrology'] - test_df['Horizontal_Distance_To_Roadways']
test_df['Roadways_Plus_Firepoints'] = test_df['Horizontal_Distance_To_Roadways'] + test_df['Horizontal_Distance_To_Fire_Points']
test_df['Roadways_Minus_Firepoints'] = test_df['Horizontal_Distance_To_Roadways'] - test_df['Horizontal_Distance_To_Fire_Points']
# Compute linear combinations of vertical distances on the test set
test_df['Elevation_Plus_Vertical_Distance_Hydrology'] = test_df['Elevation'] + test_df['Vertical_Distance_To_Hydrology']
test_df['Elevation_Minus_Vertical_Distance_Hydrology'] = test_df['Elevation'] - test_df['Vertical_Distance_To_Hydrology']
# Compute extra features for Aspect on the test set
test_df['cos_plus_sinAspect'] = np.cos((test_df['Aspect']*np.pi)/180) + np.sin((test_df['Aspect']*np.pi)/180)
# Compute linear combinations of hillshade variables
test_df['Hillshade-9_Minus_Noon'] = test_df['Hillshade_9am'] - test_df['Hillshade_Noon']
test_df['Hillshade-noon_Minus_3pm'] = test_df['Hillshade_Noon'] - test_df['Hillshade_3pm']
test_df["Hillshade-9am_Minus_3pm"] = test_df["Hillshade_9am"] - test_df["Hillshade_3pm"]

# Compute log transformations of positive features
for col in test_df.columns:
    if test_df[col].min() >= 0:
        if col == 'Cover_Type' or col.startswith('Geologic') or col.startswith('Climatic') or col.startswith('Wilderness'):
            next
        else:
            test_df['log' + col] = np.log(1+test_df[col])

# Keep the appropriate features according to the scenario (see report)

if SCENARIO=="original":
    train_df = pd.read_csv("../data/train.csv",index_col=0)
    test_df = pd.read_csv("../data/test-full.csv",index_col=0)
    selected_features =   (['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 
                       'Horizontal_Distance_To_Roadways', 
                       'Horizontal_Distance_To_Hydrology',
                       'Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_9am', 'Horizontal_Distance_To_Fire_Points', 
                        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                        "Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40",
                        'Slope'
                  ])

elif SCENARIO=="plusSoilTypeRefinement":
    selected_features =  (['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 
                       'Horizontal_Distance_To_Roadways', 
                       'Horizontal_Distance_To_Hydrology',
                       'Soil_Type','Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_9am', 'Horizontal_Distance_To_Fire_Points', 
                        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                        'ClimaticGlacial','ClimaticMontaneDry','ClimaticMontane','ClimaticMontaneDryMontane','ClimaticMontaneSubalpine','ClimaticSubalpine','ClimaticAlpine',
                        'GeologicIgneous','GeologicAlluvium','GeologicGlacial','GeologicMixedSedimentary', 'Slope'
                  ])

elif SCENARIO=="plusLinearTransformations":
    transformed_features = [
                        'Roadways_Plus_Firepoints', 'Distance_To_Hydrology',
                        'Hydrology_Plus_Roadways', 'Hydrology_Plus_Firepoints', 'Elevation_Plus_Vertical_Distance_Hydrology'
                        ,'Hydrology_Minus_Firepoints','Hydrology_Minus_Roadways',
                        'cos_plus_sinAspect'
                        ]
    selected_features =  (['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 
                       'Horizontal_Distance_To_Roadways',
                       'Horizontal_Distance_To_Hydrology',
                       'Soil_Type',
                  'Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_9am', 'Horizontal_Distance_To_Fire_Points', 
                  'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                  'ClimaticGlacial','ClimaticMontaneDry','ClimaticMontane','ClimaticMontaneDryMontane','ClimaticMontaneSubalpine','ClimaticSubalpine','ClimaticAlpine',
                  'GeologicIgneous','GeologicAlluvium','GeologicGlacial','GeologicMixedSedimentary','Slope'
                  ] + transformed_features)

elif SCENARIO=="plusDeletionHillshade":
    transformed_features = [
                        'Roadways_Plus_Firepoints', 'Distance_To_Hydrology',
                        'Hydrology_Plus_Roadways', 'Hydrology_Plus_Firepoints', 'Elevation_Plus_Vertical_Distance_Hydrology'
                        ,'Hydrology_Minus_Firepoints','Hydrology_Minus_Roadways',
                        'cos_plus_sinAspect'
                        ]

    selected_features =  (['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 
                       'Horizontal_Distance_To_Roadways',
                       'Horizontal_Distance_To_Hydrology',
                       'Soil_Type',
                   'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
                  'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                  'ClimaticGlacial','ClimaticMontaneDry','ClimaticMontane','ClimaticMontaneDryMontane','ClimaticMontaneSubalpine','ClimaticSubalpine','ClimaticAlpine',
                  'GeologicIgneous','GeologicAlluvium','GeologicGlacial','GeologicMixedSedimentary','Slope'
                  ] + transformed_features)

else:
    transformed_features = ['logHorizontal_Distance_To_Hydrology',
                        'Roadways_Plus_Firepoints', 'Distance_To_Hydrology',
                        'Hydrology_Plus_Roadways', 'Hydrology_Plus_Firepoints', 'Elevation_Plus_Vertical_Distance_Hydrology'
                        ,'Hydrology_Minus_Firepoints','Hydrology_Minus_Roadways',
                        'logSlope',
                        'cos_plus_sinAspect'
                        ]

    selected_features =  (['Elevation', 'Aspect', 'Vertical_Distance_To_Hydrology', 
                       'Horizontal_Distance_To_Roadways', 
                       'Soil_Type',
                  'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
                  'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                  'ClimaticGlacial','ClimaticMontaneDry','ClimaticMontane','ClimaticMontaneDryMontane','ClimaticMontaneSubalpine','ClimaticSubalpine','ClimaticAlpine',
                  'GeologicIgneous','GeologicAlluvium','GeologicGlacial','GeologicMixedSedimentary'
                  ] + transformed_features)


train_df = train_df[selected_features+['Cover_Type']]
test_df = test_df[selected_features]

# Save the selected features to disk
train_df.to_csv("../data/train-extra.csv",index=True)
test_df.to_csv("../data/test-extra.csv",index=True)

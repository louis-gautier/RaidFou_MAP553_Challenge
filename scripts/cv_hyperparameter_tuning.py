from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.preprocessing import StandardScaler

data_csv = "../data/train-extra.csv"
data_df = pd.read_csv(data_csv, index_col=0)
cover_type_df = data_df[["Cover_Type"]]
data_df = data_df.loc[:,data_df.columns!="Cover_Type"]
std_scaler = StandardScaler()
features_scaled = std_scaler.fit_transform(data_df.to_numpy())
data_df = pd.DataFrame(features_scaled)


model = "ExtraTrees"

if model=="ExtraTrees":
    min_samples_split = [2, 3, 5, 7, 9]
    max_features = ['auto', 'sqrt', 'log2', None]
    hyperparameter_grid = {'min_samples_split': min_samples_split,
                            'max_features': max_features}
    model = ExtraTreesClassifier()
    cv_tuning = GridSearchCV(estimator=model,
                                param_grid=hyperparameter_grid,
                                cv=5,
                                scoring = 'accuracy',
                                n_jobs = -1, verbose = 3, 
                                return_train_score = True)
    cv_tuning.fit(data_df,cover_type_df)
    print(cv_tuning.best_estimator_)

elif model=="XGBoost":
    learning_rate = [0.1,0.2,0.3,0.4,0.5]
<<<<<<< HEAD
    max_depth = [4,6,10,20]
=======
    max_depth = [4,6,10]
>>>>>>> 87c858b435175c6a1b9ce3ecf7cbf43e66305bd0
    subsample = [0.5,0.8,1]
    hyperparameter_grid = {'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample}
    model = XGBClassifier()
    cv_tuning = GridSearchCV(estimator=model,
                            param_grid=hyperparameter_grid,
                            cv=5,
                            scoring = 'accuracy',
                            n_jobs = -1, verbose = 3, 
                            return_train_score = True)
    cv_tuning.fit(data_df,cover_type_df)
    print(cv_tuning.best_estimator_)

elif model=="KNN":
    n_neighbors = [3,5,7,10,100]
    hyperparameter_grid = {'n_neighbors': n_neighbors}
    model = KNeighborsClassifier()
    cv_tuning = GridSearchCV(estimator=model,
                                param_grid=hyperparameter_grid,
                                cv=5,
                                scoring = 'accuracy',
                                n_jobs = -1, verbose = 3, 
                                return_train_score = True)
    cv_tuning.fit(data_df,cover_type_df)
    print(cv_tuning.best_estimator_)

elif model=="RandomForests":
    max_features = [None,"sqrt","log2"]
    min_samples_split = [2, 3, 5, 7, 9]
    hyperparameter_grid = {'max_features': max_features,
                           'min_samples_split': min_samples_split}
    model = RandomForestClassifier()
    cv_tuning = GridSearchCV(estimator=model,
                                param_grid=hyperparameter_grid,
                                cv=5,
                                scoring = 'accuracy',
                                n_jobs = -1, verbose = 3, 
                                return_train_score = True)
    cv_tuning.fit(data_df,cover_type_df)
    print(cv_tuning.best_estimator_)
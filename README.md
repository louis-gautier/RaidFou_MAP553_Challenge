# RaidFou_MAP553_Challenge

All the code for our project is available in the `scripts/` folder.

To compute predictions with our different models, please follow the following steps:
* First, run `add_soil_type_info.py` to perform feature engineering on soil type,
* Then, run `feature_building.py` to perform all other feature engineering steps. You can choose between the different scenarios described in appendix D by changing the SCENARIO variable to respectively "original", "plusSoilTypeRefinement", "plusLinearTransformations", "plusDeletionHillshade", "plusLogTransformation". The default value is "plusLogTransformation", implementing all our feature engineering processes.
* Finally, run `combine_methods.py` to get separate predictions for all our models as well as combined predctions. Predictions will be available in the `results/` folder.

Other useful files are:
* `cv_hyperparameter_tuning.py` and `MLP_search.py`, that were the programs that we used to tune the hyperparameters of our models.
* `data_exploration.py`, that realizes the data integrity checks and displays some basic statistics about the train and the test set.
* `trainset_oversampling.py`, that oversamples the train set by duplicating some entries while adding noise on them, so that the distribution of wilderness areas on the train set is the same than the one of the test set.

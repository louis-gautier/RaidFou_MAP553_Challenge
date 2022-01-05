from sklearn.model_selection  import RandomizedSearchCV
from sklearn.metrics          import fbeta_score, make_scorer, accuracy_score
from sklearn.model_selection  import StratifiedShuffleSplit
from sklearn.neural_network   import MLPClassifier
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#data_csv = "../data/train.csv"
#test_csv = "../data/covtype.data"
test_csv = "/Users/paultheron/Desktop/MAP553 - Projet Kaggle/map553_challenge/data/covtype.data"
data_csv = "/Users/paultheron/Desktop/MAP553 - Projet Kaggle/map553_challenge/data/train.csv"

#------------------------- Data preprocessing --------------------------
print("> Data preprocessing")

def transform( df, before, after ):
    # Note: we do different transformations depending on the sign of the feature's skew
    print("%s => %s" % (before,after))
    fudge = 1
    skew_sign = df[before].skew()
    if (skew_sign > 0):
        if (df[before].min() < 0): fudge += -(df[before].min())
        df[after] = df[before].apply(lambda x: np.log10(x+fudge))
    else:
        fudge += df[before].max()
        df[after] = df[before].apply(lambda x: np.log10(fudge-x))
        
    print("   min before:",df[before].min())
    print("   max before:",df[before].max())
    print("   skew before:", df[before].skew())
    print("   skew_after:", df[after].skew())
    print("   min after:",df[after].min())
    print("   max after:",df[after].max())
    return df

def apply_transformations( new_data ):
    new_data = transform(new_data,'hd_hy', 'hd_hy_log' )
    new_data = transform(new_data,'hd_hy_log', 'hd_hy_log_log' )
    new_data = transform(new_data,'vd_hy', 'vd_hy_log' )
    new_data = transform(new_data,'hd_rd', 'hd_rd_log' )
    new_data = transform(new_data,'hs_9',  'hs_9_log' )
    new_data = transform(new_data,'hs_9_log',  'hs_9_log_log' )
    new_data = transform(new_data,'hd_fp', 'hd_fp_log' )
    return new_data




data_df = pd.read_csv(data_csv, index_col=0)
data_df_test = pd.read_csv(test_csv, header=None)


new_cols = ['elevation','aspect','slope','hd_hy','vd_hy','hd_rd','hs_9','hs_noon','hs_3',
            'hd_fp','wa1','wa2','wa3','wa4',
            's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
            's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
            's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',
            's31','s32','s33','s34','s35','s36','s37','s38','s39','s40',
            'cover_type']

data_df.columns = new_cols


#new_data = data_df.drop(['id'],axis=1)
data_df_test.columns = new_cols
#new_data_test = data_df_test.drop(['id'],axis=1)
new_data = data_df.copy()
new_data_test = data_df_test.copy()

new_data = apply_transformations( new_data )
new_data_test = apply_transformations( new_data_test )

Drop_features = ['s7','s15','hd_hy','hd_hy_log','vd_hy','hd_rd',
                 'hs_9','hs_9_log','hd_fp','cover_type']
                 
cover_type_df = new_data[["cover_type"]]
new_data = new_data.copy().drop(Drop_features, axis=1)

cover_type_df_test = new_data_test[["cover_type"]]
new_data_test = new_data_test.copy().drop(Drop_features, axis=1)

X_train = new_data.to_numpy()
y_train = cover_type_df.to_numpy()
X_test = new_data_test.to_numpy()
y_test = cover_type_df_test.to_numpy()

print("> Classifier")

Dump_clf = True
Load_clf = False


scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # apply same transformation to test data

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(50, 19, 7), random_state=1, n_iter_no_change= 500, max_iter= 500, max_fun=25000)
print(clf)

if (Load_clf) :
    best_mlp_clf = joblib.load('best_mlp_forever.pkl')
else: # build and dump new classifier
    parameters = {'hidden_layer_sizes': [(50,25,7), (50,100,7), (100,7), (50,7)],
                    'activation': ['tanh', 'relu'],
                    'batch_size': [16, 64, 128, 256, 512],
                    'solver': ['sgd', 'adam', 'lbfgs'],
                    'learning_rate': ['constant','adaptive'],
                    'alpha': np.array([10.0, 5.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
                    'learning_rate_init': np.array([1.0, 0.1, 0.01, 0.001]),
                 }

    scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')

    #  Use a stratified sample because there are a lot more examples of one class that the other in the input data
    #cv = StratifiedShuffleSplit(y_train, test_size=0.2, random_state=42)

    #grid_obj = GridSearchCV(clf, param_grid=parameters, cv=5, scoring='accuracy', verbose=3, n_jobs=-1, return_train_score = True)    
    grid_obj = RandomizedSearchCV(clf, param_distributions=parameters, cv=5, scoring='accuracy', verbose=3, n_jobs=-1, return_train_score = True)    


    grid_fit = grid_obj.fit(X_train_scaled, np.ravel(y_train))

    # Get the estimator
    best_mlp_clf    = grid_obj.best_estimator_
    print("best params found by RandomizedSearchCV=%s" % grid_obj.best_params_)
    # best_params = grid_obj.best_params_
    if (Dump_clf): joblib.dump(best_mlp_clf, 'best_mlp_forever.pkl')

best_name   = best_mlp_clf.__class__.__name__
best_params = best_mlp_clf.get_params()

print("Best classifier is %s" % best_name)
print("Best params=%s\n"      % best_params)

# Make predictions using the unoptimized and optimized models
train_predictions = best_mlp_clf.predict(X_train_scaled)
test_predictions  = (clf.fit(X_train_scaled, np.ravel(y_train))).predict(X_test_scaled)
best_predictions  = best_mlp_clf.predict(X_test_scaled)

# Report the before-and-afterscores

print("Final accuracy score on the training data: {:.4f}".format(accuracy_score(y_train, train_predictions)))
print("Final F-score on the training data: {:.4f}".format(fbeta_score(y_train, train_predictions, beta = 0.5, average='weighted')))

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, test_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, test_predictions, beta = 0.5, average='weighted')))

print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='weighted')))

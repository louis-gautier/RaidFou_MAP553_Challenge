from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import joblib


data_csv = "../data/"
out_file = "../results/"
secret_test_file = "../data/"

RECOMPUTE_MODEL = True

#------------------------------ Utils Function------------------------------------------


def compute_acc_no_test(y_pred, y_probas, model_1_2):
  predictions = [round(value) for value in y_pred]
  predictions_inf_2 = [x <= 2 for x in predictions]
  X_1_2 = X_val.loc[predictions_inf_2,:]
  if not (X_1_2.empty):
    final_predictions_probas_inf_2 = model_1_2.predict_proba(X_1_2)
    idx_inf_2=0
    for i in range(len(predictions)):
      if predictions_inf_2[i]:
        proba1 = y_probas[i][0] + final_predictions_probas_inf_2[idx_inf_2][0]/1.5
        proba2 = y_probas[i][1] + final_predictions_probas_inf_2[idx_inf_2][1]
        if proba1>proba2:
          predictions[i] = 1
        else:
          predictions[i] = 2
        idx_inf_2 += 1
  accuracy_val = accuracy_score(y_val, predictions)
  return accuracy_val, predictions

def compute_acc_no_test_on_train(y_pred, y_probas, model_1_2):
  predictions = [round(value) for value in y_pred]
  predictions_inf_2 = [x <= 2 for x in predictions]
  X_1_2 = X_train.loc[predictions_inf_2,:]
  if not (X_1_2.empty): 
    final_predictions_probas_inf_2 = model_1_2.predict_proba(X_1_2)
    idx_inf_2=0
    for i in range(len(predictions)):
      if predictions_inf_2[i]:
        proba1 = y_probas[i][0] + final_predictions_probas_inf_2[idx_inf_2][0]/1.5
        proba2 = y_probas[i][1] + final_predictions_probas_inf_2[idx_inf_2][1]
        if proba1>proba2:
          predictions[i] = 1
        else:
          predictions[i] = 2
        idx_inf_2 += 1
  accuracy_val = accuracy_score(y_train, predictions)
  return accuracy_val, predictions

def compute_acc_with_test(y_pred, y_probas, model_1_2):
  predictions = [round(value) for value in y_pred]
  predictions_inf_2 = [x <= 2 for x in predictions]
  X_1_2 = X_test.loc[predictions_inf_2,:]
  if not (X_1_2.empty):
    final_predictions_probas_inf_2 = model_1_2.predict_proba(X_1_2)
    idx_inf_2=0
    for i in range(len(predictions)):
      if predictions_inf_2[i]:
        proba1 = y_probas[i][0] + final_predictions_probas_inf_2[idx_inf_2][0]/1.5
        proba2 = y_probas[i][1] + final_predictions_probas_inf_2[idx_inf_2][1]
        if proba1>proba2:
          predictions[i] = 1
        else:
          predictions[i] = 2
        idx_inf_2 += 1
    test_accuracy = accuracy_score(secret_labels, predictions)
  return test_accuracy, predictions

def compute_acc_with_test_combin(y_pred, y_probas, model_1_2):
  predictions = [round(value) for value in y_pred]
  predictions_inf_2 = [x <= 2 for x in predictions]
  X_1_2 = test_set.loc[predictions_inf_2,:]
  if not (X_1_2.empty):
    final_predictions_probas_inf_2 = model_1_2.predict_proba(X_1_2)
    idx_inf_2=0
    for i in range(len(predictions)):
      if predictions_inf_2[i]:
        proba1 = y_probas[i][0] + final_predictions_probas_inf_2[idx_inf_2][0]/1.5
        proba2 = y_probas[i][1] + final_predictions_probas_inf_2[idx_inf_2][1]
        if proba1>proba2:
          predictions[i] = 1
        else:
          predictions[i] = 2
        idx_inf_2 += 1
  test_accuracy = accuracy_score(secret_labels, predictions)
  return test_accuracy, predictions
  
#------------------------------ Prepare Data ------------------------------------------
print("> Preparing Data")
data_df = pd.read_csv(data_csv+ "train-extra.csv", index_col=0)
cover_type_df = data_df[["Cover_Type"]]
data_df = data_df.loc[:,data_df.columns!="Cover_Type"]
std_scaler = StandardScaler()
features_scaled = std_scaler.fit_transform(data_df.to_numpy())
data_df = pd.DataFrame(features_scaled)
seed = 1
test_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(data_df, cover_type_df, test_size=test_size, random_state=seed)
train_entries_1_2 = (y_train['Cover_Type']<=2).to_numpy()
train_df_1_2 = X_train.loc[train_entries_1_2,:]
labels_df_1_2 = y_train.loc[train_entries_1_2,:]

X_test = pd.read_csv(data_csv + "test-extra.csv",index_col=0)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
features_scaled = std_scaler.transform(X_test.to_numpy())
X_test = pd.DataFrame(features_scaled)

secret_test = pd.read_csv(secret_test_file + "covtype.data",header=None)
secret_labels = secret_test.loc[:,54].values

#------------------------------ Models ------------------------------------------
if RECOMPUTE_MODEL:
  print("> Recomputing models, saving in results file")

  print("> Fitting Models")

  print("   > Fitting Extra Tree")

  model_XTREE = ExtraTreesClassifier(n_estimators=1000,min_samples_split=3,max_features=None, random_state=42, n_jobs=-1)
  model_XTREE.fit(X_train, np.ravel(y_train))
  model_XTREE_1_2 = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1)
  model_XTREE_1_2.fit(train_df_1_2,np.ravel(labels_df_1_2))

  print("   > Fitting Random Forest")

  model_RANDFO = RandomForestClassifier(n_estimators=1000,min_samples_split=3,max_features="sqrt",class_weight='balanced',n_jobs=-1,random_state=42)
  model_RANDFO.fit(X_train, np.ravel(y_train))
  model_RANDFO_1_2 = RandomForestClassifier(n_estimators=1000,min_samples_split=3,max_features="sqrt",class_weight='balanced',n_jobs=-1,random_state=42)
  model_RANDFO_1_2.fit(train_df_1_2,np.ravel(labels_df_1_2))

  print("   > Fitting SVC")

  model_SVC = SVC(C=10, gamma = 0.1, probability=True, cache_size=2000)
  model_SVC.fit(X_train, np.ravel(y_train))
  model_SVC_1_2 = SVC(C=10, gamma = 0.1, probability=True, cache_size= 2000)
  model_SVC_1_2.fit(train_df_1_2,np.ravel(labels_df_1_2))

  print("   > Fitting MLPClassifier")

  model_MLP = MLPClassifier(activation='tanh', alpha=5.0, batch_size=7, hidden_layer_sizes=(50, 100, 7), learning_rate='adaptive', learning_rate_init=1.0, solver='lbfgs', random_state=1, n_iter_no_change= 500, max_iter= 500, max_fun=25000)
  model_MLP.fit(X_train, np.ravel(y_train))
  model_MLP_1_2 = MLPClassifier(activation='tanh', alpha=5.0, batch_size=7, hidden_layer_sizes=(50, 100, 7), learning_rate='adaptive', learning_rate_init=1.0, solver='lbfgs', random_state=1, n_iter_no_change= 500, max_iter= 500, max_fun=25000)
  model_MLP_1_2.fit(train_df_1_2,np.ravel(labels_df_1_2))

  #------------------------------ Predictions ------------------------------------------

  print("> Predicting on validation")

  print("   > Predicting Extra Tree")

  # make predictions for test data
  y_pred_XTREE = model_XTREE.predict(X_val)
  y_probas_XTREE = model_XTREE.predict_proba(X_val)
  accuracy_XTREE_val, predictions_XTREE = compute_acc_no_test(y_pred_XTREE, y_probas_XTREE, model_XTREE_1_2)
  print("     > Accuracy XTREE validation: %.2f%%" % (accuracy_XTREE_val * 100.0))
  #print(confusion_matrix(y_val,predictions_XTREE))

  print("   > Predicting Random Forest")

  # make predictions for test data
  y_pred_RANDFO = model_RANDFO.predict(X_val)
  y_probas_RANDFO = model_RANDFO.predict_proba(X_val)
  accuracy_RANDFO_val, predictions_RANDFO = compute_acc_no_test(y_pred_RANDFO, y_probas_RANDFO, model_RANDFO_1_2)
  print("     > Accuracy RANDFO validation: %.2f%%" % (accuracy_RANDFO_val * 100.0))

  print("   > Predicting SVC")

  # make predictions for test data
  y_pred_SVC = model_SVC.predict(X_val)
  y_probas_SVC = model_SVC.predict_proba(X_val)
  accuracy_SVC_val, predictions_SVC = compute_acc_no_test(y_pred_SVC, y_probas_SVC, model_SVC_1_2)

  print("     > Accuracy SVC validation: %.2f%%" % (accuracy_SVC_val * 100.0))
  #print(confusion_matrix(y_val,predictions_SVC))

  print("   > Predicting MLPClassifier")

  # make predictions for test data
  y_pred_MLP = model_MLP.predict(X_val)
  y_probas_MLP = model_MLP.predict_proba(X_val)
  accuracy_MLP_val, predictions_MLP = compute_acc_no_test(y_pred_MLP, y_probas_MLP, model_MLP_1_2)

  print("     > Accuracy MLP validation: %.2f%%" % (accuracy_MLP_val * 100.0))
  #print(confusion_matrix(y_val,predictions_MLP))

  # ----

  print("> Predicting on testing")


  print("   > Predicting Extra Tree")

  y_pred_XTREE = model_XTREE.predict(X_test)
  y_probas_XTREE = model_XTREE.predict_proba(X_test)
  test_accuracy_XTREE, predictions_XTREE = compute_acc_with_test(y_pred_XTREE, y_probas_XTREE, model_XTREE_1_2)
  print("     > Accuracy XTREE testing: %.4f%%" % (test_accuracy_XTREE * 100.0))

  print("   > Predicting RANDFO")

  y_pred_RANDFO = model_RANDFO.predict(X_test)
  y_probas_RANDFO = model_RANDFO.predict_proba(X_test)
  test_accuracy_RANDFO, predictions_RANDFO = compute_acc_with_test(y_pred_RANDFO, y_probas_RANDFO, model_RANDFO_1_2)
  print("     > Accuracy RANDFO testing: %.4f%%" % (test_accuracy_RANDFO * 100.0))

  print("   > Predicting SVC")

  y_pred_SVC = model_SVC.predict(X_test)
  y_probas_SVC = model_SVC.predict_proba(X_test)
  test_accuracy_SVC, predictions_SVC = compute_acc_with_test(y_pred_SVC, y_probas_SVC, model_SVC_1_2)
  print("     > Accuracy SVC testing: %.4f%%" % (test_accuracy_SVC * 100.0))


  print("   > Predicting MLP")

  y_pred_MLP = model_MLP.predict(X_test)
  y_probas_MLP = model_MLP.predict_proba(X_test)
  test_accuracy_MLP, predictions_MLP = compute_acc_with_test(y_pred_MLP, y_probas_MLP, model_MLP_1_2)
  print("     > Accuracy MLP testing: %.4f%%" % (test_accuracy_MLP * 100.0))


  #------------------------------ Saving Results Intermediary ------------------------------------------

  print("> Saving Intermediary Results")

  print("   > Saving Extra Tree")
  output_XTREE = pd.DataFrame(predictions_XTREE)
  #print(confusion_matrix(secret_labels,predictions_XTREE))
  output_XTREE.index+=1
  output_XTREE.to_csv(out_file + "predictions_XTREE.csv" , index_label="Id")
  joblib.dump(model_XTREE, out_file + "model_XTREE"+ ".model")
  joblib.dump(model_XTREE_1_2, out_file + "model_XTREE_1_2" + ".model")

  print("   > Saving Randforest")
  output_RANDFO = pd.DataFrame(predictions_RANDFO)
  #print(confusion_matrix(secret_labels,predictions_RANDFO))
  output_RANDFO.index+=1
  output_RANDFO.to_csv(out_file + "predictions_RANDFO.csv" , index_label="Id")
  joblib.dump(model_RANDFO, out_file + "model_RANDFO"+ ".model")
  joblib.dump(model_RANDFO_1_2, out_file + "model_RANDFO_1_2" + ".model")


  print("   > Saving SVC")
  output_SVC = pd.DataFrame(predictions_SVC)
  #print(confusion_matrix(secret_labels,predictions_SVC))
  output_SVC.index+=1
  output_SVC.to_csv(out_file + "predictions_SVC.csv" , index_label="Id")
  joblib.dump(model_SVC, out_file + "model_SVC"+ ".model")
  joblib.dump(model_SVC_1_2, out_file + "model_SVC_1_2"+ ".model")

  print("   > Saving MLP")
  output_MLP = pd.DataFrame(predictions_MLP)
  #print(confusion_matrix(secret_labels,predictions_MLP))
  output_MLP.index+=1
  output_MLP.to_csv(out_file + "predictions_MLP.csv" , index_label="Id")
  joblib.dump(model_MLP, out_file + "model_MLP" + ".model")
  joblib.dump(model_MLP_1_2, out_file + "model_MLP_1_2" + ".model")


#------------------------------ Logistic regression ------------------------------------------
else:
  print("> Loading models from results file")
  model_XTREE = joblib.load(out_file + "model_XTREE.model")
  model_XTREE_1_2 = joblib.load(out_file + "model_XTREE_1_2.model")
  model_RANDFO = joblib.load(out_file + "model_RANDFO.model")
  model_RANDFO_1_2 = joblib.load(out_file + "model_RANDFO_1_2.model")
  model_SVC = joblib.load(out_file + "model_SVC.model")
  model_SVC_1_2 = joblib.load(out_file + "model_SVC_1_2.model")
  model_MLP = joblib.load(out_file + "model_MLP.model")
  model_MLP_1_2 = joblib.load(out_file + "model_MLP_1_2.model")
  print("> Loading predictions from results file")
  predictions_XTREE = pd.read_csv(out_file + "predictions_XTREE.csv", index_col=0).to_numpy()
  predictions_RANDFO = pd.read_csv(out_file + "predictions_RANDFO.csv", index_col=0).to_numpy()
  predictions_SVC = pd.read_csv(out_file + "predictions_SVC.csv", index_col=0).to_numpy()
  predictions_MLP = pd.read_csv(out_file + "predictions_MLP.csv", index_col=0).to_numpy()


print("> Computing Logistic regression")

print("   > Preparing Data")

print("      > Predicting Extra Tree on train set")

y_pred_XTREE_train = model_XTREE.predict(X_train)
y_probas_XTREE_train = model_XTREE.predict_proba(X_train)
accuracy_XTREE_train, predictions_XTREE_train = compute_acc_no_test_on_train(y_pred_XTREE_train, y_probas_XTREE_train, model_XTREE_1_2)

print("      > Predicting RANDFO on train set")

y_pred_RANDFO_train = model_RANDFO.predict(X_train)
y_probas_RANDFO_train = model_RANDFO.predict_proba(X_train)
accuracy_RANDFO_train, predictions_RANDFO_train = compute_acc_no_test_on_train(y_pred_RANDFO_train, y_probas_RANDFO_train, model_RANDFO_1_2)

print("      > Predicting SVC on train set")

y_pred_SVC_train = model_SVC.predict(X_train)
y_probas_SVC_train = model_SVC.predict_proba(X_train)
accuracy_SVC_train, predictions_SVC_train = compute_acc_no_test_on_train(y_pred_SVC_train, y_probas_SVC_train, model_SVC_1_2)

print("      > Predicting MLPClassifier on train set")

y_pred_MLP_train = model_MLP.predict(X_train)
y_probas_MLP_train = model_MLP.predict_proba(X_train)
accuracy_MLP_train, predictions_MLP_train = compute_acc_no_test_on_train(y_pred_MLP_train, y_probas_MLP_train, model_MLP_1_2)

train_set = pd.DataFrame(np.transpose([predictions_XTREE_train, predictions_RANDFO_train,predictions_SVC_train, predictions_MLP_train]))
train_set_1_2 = train_set.loc[train_entries_1_2,:]
test_set = pd.DataFrame(np.transpose([np.ravel(predictions_XTREE), np.ravel(predictions_RANDFO), np.ravel(predictions_SVC), np.ravel(predictions_MLP)]))

'''
count = 0
count2 =0
for ind, (a,b,c) in enumerate(np.transpose([predictions_XTREE, predictions_SVC, predictions_MLP])[0]):
  if not (a==b==c) : 
    count +=1
    if not (a == secret_labels[ind]):
      count2 +=1
      print("XT, SVM, MLP",(a,b,c))
      print("Real", secret_labels[ind])
      print(" ")
    
print(count)
print(count2)'''


print("   > Fitting Combinator")

def majority_vote(x):
  label_0 = x[0]
  label_1 = x[1]
  label_2 = x[2]
  if label_0 == label_1:
    return label_1
  elif label_0 == label_2:
    return label_0
  elif label_1 == label_2:
    return label_1
  else:
    return label_0



combinator = LogisticRegression()
combinator_1_2 = LogisticRegression()

#combinator = SGDClassifier(loss='log')
#combinator_1_2 = SGDClassifier(loss='log')

combinator.fit(train_set, np.ravel(y_train))
combinator_1_2.fit(train_set_1_2,np.ravel(labels_df_1_2))

print("   > Predicting on testing")

y_pred_combinator = combinator.predict(test_set)
y_probas_combinator = combinator.predict_proba(test_set)
test_accuracy_combinator, predictions_combinator = compute_acc_with_test_combin(y_pred_combinator, y_probas_combinator, combinator_1_2)
#y_pred_combinator = [majority_vote(x) for x in test_set.to_numpy()]

test_accuracy_combinator = accuracy_score(secret_labels, y_pred_combinator)
print("      > Accuracy Combination testing: %.4f%%" % (test_accuracy_combinator * 100.0))
print(confusion_matrix(secret_labels,y_pred_combinator))

print("   > Saving Intermediary Results")

output_combinator = pd.DataFrame(y_pred_combinator)
output_combinator.index+=1
output_combinator.to_csv(out_file + "predictions_combinator.csv" , index_label="Id")
import pandas as pd
import numpy as np
from numpy import random as rd
import scipy.optimize as optimize

# Read the train set
train_df = pd.read_csv("../data/train-extra.csv",index_col=0)
# Compute the standard deviation of each feature to add noise later
stds = train_df.std(axis=0)
# Partition the trainset according to the wilderness area
train_wa1 = train_df.loc[train_df.Wilderness_Area1==1,:]
train_wa2 = train_df.loc[train_df.Wilderness_Area2==1,:]
train_wa3 = train_df.loc[train_df.Wilderness_Area3==1,:]
train_wa4 = train_df.loc[train_df.Wilderness_Area4==1,:]
# Count the number of entries in each wilderness area
nb1 = len(train_wa1.index)
nb2 = len(train_wa2.index)
nb3 = len(train_wa3.index)
nb4 = len(train_wa4.index)
# Compute the rate of each wilderness area in the test set 
test_df = pd.read_csv("../data/test-extra.csv",index_col=0)
test_wa1 = test_df.loc[test_df.Wilderness_Area1==1,:]
test_wa2 = test_df.loc[test_df.Wilderness_Area2==1,:]
test_wa3 = test_df.loc[test_df.Wilderness_Area3==1,:]
test_wa4 = test_df.loc[test_df.Wilderness_Area4==1,:]
target1 = len(test_wa1.index)/len(test_df.index)
target2 = len(test_wa2.index)/len(test_df.index)
target3 = len(test_wa3.index)/len(test_df.index)
target4 = len(test_wa4.index)/len(test_df.index)

# Compute the number of points to add in wilderness areas 1,2 and 3 to come as close as possible to the rates computed above (linear optimization under linear constraints)
A = np.array([[1/target1-1,-1,-1],[-1,1/target2-1,-1],[-1,-1,1/target3-1]])
B = np.array([nb1+nb2+nb3+nb4-nb1/target1,nb1+nb2+nb3+nb4-nb2/target2,nb1+nb2+nb3+nb4-nb3/target3])
def f(x):
    y = np.dot(A, x) - B
    return np.dot(y,y)

cons = ({'type': 'ineq', 'fun': lambda x: x[0]},{'type': 'ineq', 'fun': lambda x: x[1]},{'type': 'ineq', 'fun': lambda x: x[2]})
res = optimize.minimize(f, [0, 0, 0], method='COBYLA', constraints=cons, options={'maxiter':10000000})
xbest = res['x']

# Print the achieved rates once the points will be added
total = int(xbest[0])+int(xbest[1])+int(xbest[2])+nb1+nb2+nb3+nb4
print((nb1+xbest[0])/total)
print((nb2+xbest[1])/total)
print((nb3+xbest[2])/total)
print((nb4)/total)

# Add new points to wilderness areas 1,2 and 3

for i in range(int(xbest[0])):
  # Take a new point at random
  row = train_wa1.sample()
  for col in row.columns:
    # For each non-categorical column, add a random noise with a variance of 10% of the variance of the feature
    if not(col.startswith("Climatic")) and not(col.startswith("Geologic")) and not (col.startswith("Soil")) and not(col.startswith("Cover_Type")):
      row.loc[:,col] += rd.normal()*(1/10)*stds[col]
  train_df = train_df.append(row,ignore_index=True)

for i in range(int(xbest[1])):
  row = train_wa2.sample()
  for col in row.columns:
    if not(col.startswith("Climatic")) and not(col.startswith("Geologic")) and not (col.startswith("Soil")) and not(col.startswith("Cover_Type")):
      row.loc[:,col] += rd.normal()*(1/10)*stds[col]
  train_df = train_df.append(row,ignore_index=True)

for i in range(int(xbest[2])):
  row = train_wa3.sample()
  for col in row.columns:
    if not(col.startswith("Climatic")) and not(col.startswith("Geologic")) and not (col.startswith("Soil")) and not(col.startswith("Cover_Type")):
      row.loc[:,col] += rd.normal()*(1/10)*stds[col]
  train_df = train_df.append(row,ignore_index=True)

# Write the oversampled train set to disk
train_df.to_csv("drive/MyDrive/data553/train-extra.csv",index=True)
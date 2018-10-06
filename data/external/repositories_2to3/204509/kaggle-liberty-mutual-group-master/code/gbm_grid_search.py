# -*- coding: utf-8 -*-

"""
code to find best parameters for gradient boosting machine from sklearn.
"""

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn import preprocessing

def Gini(y_true, y_pred):
    n_samples = y_true.shape[0]
    
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    return G_pred/G_true


print('Reading data...')
train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)
test_ind = test.index
train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

loss_range = np.array(['huber'])
learning_rate_range = np.array([0.05, 0.057, 0.065])
n_estimators_range = np.array([100])
max_features_range = np.array(['auto', 'sqrt', 'log2'])
max_depth_range = np.array([3, 4, 6])
min_samples_split_range = np.array([2, 3, 5])
min_samples_leaf_range = np.array([4, 6, 8])
alpha_range = np.array([0.8, 0.85, 0.9])
max_leaf_nodes_range = np.array([10, 20, 30])
min_weight_fraction_leaf_range = np.array([0.001, 0.003, 0.005])

print("Grid search...")
param_grid = dict(loss=loss_range.tolist(),
                  learning_rate=learning_rate_range.tolist(),
                  n_estimators=n_estimators_range.tolist(), 
                  max_features=max_features_range.tolist(), 
                  max_depth=max_depth_range.tolist(), 
                min_samples_split=min_samples_split_range.tolist(),
                min_samples_leaf=min_samples_leaf_range.tolist(),
                alpha=alpha_range.tolist(),
                max_leaf_nodes=max_leaf_nodes_range.tolist(),
                min_weight_fraction_leaf=min_weight_fraction_leaf_range.tolist())                


rgr = GradientBoostingRegressor()
scorer = make_scorer(Gini)
grid = GridSearchCV(rgr, param_grid, scoring=scorer)
grid.fit(train, labels)
print(("The best classifier is: ", grid.best_estimator_))

# -*- coding: utf-8 -*-

"""
code to make a prediction by xgboost.
it can work in test mode if you change TEST to 1.
number of iteration in result can be changed.
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

TEST = 0 # or make a submission file
ITER = 5 # number of iterations

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
    
    
def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 9
        
    plst = list(params.items())
     
    data_train, data_val, labels_train, labels_val = train_test_split(train, labels, test_size=.2)
    
    num_rounds = 10000    
    
    xgtest = xgb.DMatrix(test)
    
    xgtrain = xgb.DMatrix(data_train, label=labels_train)
    xgval = xgb.DMatrix(data_val, label=labels_val)
    
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)
    
    data_train, data_val, labels_train, labels_val = train_test_split(train, labels, test_size=.2)
    labels_train = np.log(labels_train)
    labels_val = np.log(labels_val)
    
    xgtrain = xgb.DMatrix(data_train, label=labels_train)
    xgval = xgb.DMatrix(data_val, label=labels_val)
    
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)
    
    preds = preds1*1.4 + preds2*8.6
    return preds


print('Reading data...')

if (TEST):
    data = pd.read_csv('../input/train.csv', index_col=0)
    labels_data = data.Hazard
    data.drop('Hazard', axis=1, inplace=True)
    train, test, labels, labels_test = train_test_split(data, labels_data, test_size=.2)
else:
    train = pd.read_csv('../input/train.csv', index_col=0)
    test = pd.read_csv('../input/test.csv', index_col=0)
        
    labels = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    
train_s = train
test_s = test

train_s.drop('T2_V10', axis=1, inplace=True)
train_s.drop('T2_V7', axis=1, inplace=True)
train_s.drop('T1_V13', axis=1, inplace=True)
train_s.drop('T1_V10', axis=1, inplace=True)

test_s.drop('T2_V10', axis=1, inplace=True)
test_s.drop('T2_V7', axis=1, inplace=True)
test_s.drop('T1_V13', axis=1, inplace=True)
test_s.drop('T1_V10', axis=1, inplace=True)

columns = train.columns
test_ind = test.index
train_s = np.array(train_s)
test_s = np.array(test_s)

for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)

train = list(train.T.to_dict().values())
test = list(test.T.to_dict().values())

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

if (TEST):
    preds1 = xgboost_pred(train_s, labels, test_s)
    preds2 = xgboost_pred(train, labels, test)
    preds = (0.47 * (preds1 ** 0.045) + 0.53 * (preds2 ** 0.055))
    
    print(('Validation Score: ', Gini(labels_test, preds)))

else:
    preds = np.zeros(test.shape[0])
    
    for iter in range(ITER):
        preds1 = xgboost_pred(train_s, labels, test_s)
        preds2 = xgboost_pred(train, labels, test)
        preds += (0.47 * (preds1 ** 0.045) + 0.53 * (preds2 ** 0.055)) / ITER
    
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv('../output/xgboost_1.csv')
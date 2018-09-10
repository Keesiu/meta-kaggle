#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python

import pandas as pd
from os import listdir
from sklearn.metrics import log_loss
from helperFunctions import *
from classes import *

target = pd.read_csv("../data/train.csv").target
ID = pd.read_csv("../data/test.csv").id
SEED = 45

train = {}
test = {}
for f in [x for x in listdir("../data/") if x.startswith("train_pred")]:
	train[f] = pd.read_csv("../data/"+f).values
	test[f] = pd.read_csv("../data/"+f.replace('train','test'))
	test[f] = test[f].drop('id', axis=1)
	test[f] = test[f].values

weights = ensembleModels(train,target,10,45)
finalTrainPred = pd.DataFrame(applyWeights(train,weights),columns=target.unique())
finalTestPred = pd.DataFrame(applyWeights(test,weights),columns=target.unique())
finalTestPred['id'] = ID
finalTestPred = finalTestPred[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]
print "\tConfirmed Log Loss:",log_loss(target,finalTrainPred.values)

## Save Results ##
finalTestPred.to_csv("../data/test_pred_blend.csv",index=False)



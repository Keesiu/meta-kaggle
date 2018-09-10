#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python
"""
Runs a solution for the Otto Group product categorization Kaggle challenge. Example run:

./train.py --k 5 --log none --pca 0 --rfe 45 --seed 52 > ../data/results_5_none_0_45_52.txt
"""


import argparse
import pandas as pd
import numpy as np
from joblib import Parallel, delayed  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from helperFunctions import *
from classes import *

#################
## Parse Input ##
#################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--path", dest="path", type=str,default="../data/")
parser.add_argument("--k", dest="k", type=int,default=5)
parser.add_argument("--log", dest="log", choices=["add", "replace", "none"],default="add")
parser.add_argument("--pca", dest="pca_n", type=int,default=0)
parser.add_argument("--rfe", dest="rfe", type=int,default=0)
parser.add_argument("--seed", dest="SEED", type=int,default=34)

args = parser.parse_args()

##########
## Main ##
##########

def main(path="../data/",k=5,log='add',pca_n=0,rfe=45,SEED=34):
	## Load Data ##
	train, test, target, target_nnet, id, cv, encoder = loadData(path,k,log,pca_n,SEED)
	
	## Run Feature Selection ##
	if rfe > 0:
		rfeFeat = Parallel(n_jobs=-1)(delayed(featSelect)(label,train,target.apply(lambda x: 1 if x == label else 0).values,cv,rfe,SEED,label) for label in target.unique())
		rfeFeat = listToDict(rfeFeat)
	else:
		rfeFeat = {}
		for label in target.unique():
			rfeFeat[label] = train.columns.values
	
	print rfeFeat

	## Independent Class Models ##
	rfGrid = {
		'Class_1': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_2': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_3': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_4': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_5': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_6': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_7': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_8': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
		'Class_9': {
			'model': RandomForestClassifier(n_jobs=2,random_state=SEED),
			'grid': {
				'n_estimators':[300],
				'max_depth':[35,40],
				'min_samples_split':[2],
				'min_samples_leaf':[2],
				'max_features':[10,15]
			}},
	}
	gbmGrid = {
		'Class_1': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.11],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_2': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.1],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_3': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.1],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_4': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.1],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_5': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.11],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_6': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.08],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_7': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.11],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
		'Class_8': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.1],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[12]
			}},
		'Class_9': {
			'model': GradientBoostingClassifier(random_state=SEED),
			'grid': {
				'n_estimators':[200],
				'learning_rate':[0.08],
				'max_depth':[30],
				'min_samples_split':[2],
				'min_samples_leaf':[5],
				'max_features':[10]
			}},
	}
	glmGrid = {
		'Class_1': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_2': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_3': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_4': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_5': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_6': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_7': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_8': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
		'Class_9': {
			'model': LogisticRegression(random_state=SEED),
			'grid': {
				'C':[0.5]
			}},
	}
	svmGrid = {
		'Class_1': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_2': {
			'model': SVC(random_state=SEED,probability=True,max_iter=50000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_3': {
			'model': SVC(random_state=SEED,probability=True,max_iter=50000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_4': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_5': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_6': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_7': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_8': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
		'Class_9': {
			'model': SVC(random_state=SEED,probability=True,max_iter=100000),
			'grid': {
				  'C':[5],
				  'kernel':['rbf'],
				  'degree': [2],
				  'gamma':[0.0],
				  'shrinking':[False]
			}},
	}
	nnetRawGrid = {
		'Class_1': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [20]
			}},
		'Class_2': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [25]
			}},
		'Class_3': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_4': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_5': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_6': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_7': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_8': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
		'Class_9': {
			'model': NeuralNet(layers=[
						('input', InputLayer),
						('dropoutf', DropoutLayer),
						('dense0', DenseLayer),
						('dropout', DropoutLayer),
						('dense1', DenseLayer),
						('dropout2', DropoutLayer), 
						('output', DenseLayer)
					],
					input_shape=(None,len([x for x in train.columns if x.startswith('feat_')])),
					output_num_units=2,
					output_nonlinearity=softmax,
					update=nesterov_momentum,
					verbose=0),
			'grid': {
				'dropoutf_p':[0.15],
				'dense0_num_units':[500],
				'dropout_p':[0.25],
				'dense1_num_units':[250],
				'dropout2_p':[0.35],
				'eval_size':[0.05],
				'update_learning_rate': [theano.shared(float32(0.01))],
				'update_momentum': [theano.shared(float32(0.9))],
				'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
				'max_epochs': [30]
			}},
	}

	nnetRawModel = Parallel(n_jobs=-1)(delayed(trainModel)(label,SingleModel(label+' nnet raw',nnetRawGrid[label]['model']),target.apply(lambda x: 1 if x == label else 0).values.astype('int32'),train[[x for x in train.columns if x.startswith('feat_')]],test[[x for x in train.columns if x.startswith('feat_')]],nnetRawGrid[label]['grid'],cv,'logloss') for label in target.unique())
	for l in nnetRawModel:
		train[l.keys()[0]+'_nnet_raw_pred'] = l[l.keys()[0]]['pred']
		test[l.keys()[0]+'_nnet_raw_pred'] = l[l.keys()[0]]['test_pred']

	rfModel = Parallel(n_jobs=-1)(delayed(trainModel)(label,SingleModel(label+' rf',rfGrid[label]['model']),target.apply(lambda x: 1 if x == label else 0).values,train[rfeFeat[label]],test[rfeFeat[label]],rfGrid[label]['grid'],cv,'auc') for label in target.unique())
	for l in rfModel:
		train[l.keys()[0]+'_rf_pred'] = l[l.keys()[0]]['pred']
		test[l.keys()[0]+'_rf_pred'] = l[l.keys()[0]]['test_pred']

	gbmModel = Parallel(n_jobs=-1)(delayed(trainModel)(label,SingleModel(label+' gbm',gbmGrid[label]['model']),target.apply(lambda x: 1 if x == label else 0).values,train[rfeFeat[label]],test[rfeFeat[label]],gbmGrid[label]['grid'],cv,'auc') for label in target.unique())
	for l in gbmModel:
		train[l.keys()[0]+'_gbm_pred'] = l[l.keys()[0]]['pred']
		test[l.keys()[0]+'_gbm_pred'] = l[l.keys()[0]]['test_pred']

	glmModel = Parallel(n_jobs=-1)(delayed(trainModel)(label,SingleModel(label+' glm',glmGrid[label]['model']),target.apply(lambda x: 1 if x == label else 0).values,train[rfeFeat[label]],test[rfeFeat[label]],glmGrid[label]['grid'],cv,'auc') for label in target.unique())
	for l in glmModel:
		train[l.keys()[0]+'_glm_pred'] = l[l.keys()[0]]['pred']
		test[l.keys()[0]+'_glm_pred'] = l[l.keys()[0]]['test_pred']

	svmModel = Parallel(n_jobs=-1)(delayed(trainModel)(label,SingleModel(label+' svm',svmGrid[label]['model']),target.apply(lambda x: 1 if x == label else 0).values,train[rfeFeat[label]],test[rfeFeat[label]],svmGrid[label]['grid'],cv,'auc') for label in target.unique())
	for l in svmModel:
		train[l.keys()[0]+'_svm_pred'] = l[l.keys()[0]]['pred']
		test[l.keys()[0]+'_svm_pred'] = l[l.keys()[0]]['test_pred']

	for label in target.unique():
		tmp = train[[v for v in train.columns if v.startswith(label)]].to_dict('series')
		for x in tmp:
			tmp[x] = tmp[x].values
		weights = ensembleModels(tmp,target.apply(lambda x: 1 if x == label else 0),10,SEED)
		train[label+'_ensemble_pred'] = applyWeights(train[[x for x in weights.keys()]],weights)
		test[label+'_ensemble_pred'] = applyWeights(test[[x for x in weights.keys()]],weights)
		
	## Multi-Class Models ##
	trainPred = {}
	testPred = {}
	rfMultiGrid = {
		'model': RandomForestClassifier(n_jobs=-1,random_state=SEED),
		'grid': {
			  'n_estimators':[500,750],
			  'max_depth':[20],
			  'min_samples_split':[2],
			  'min_samples_leaf':[2],
			  'max_features':[8]
		},
		'vars': [x for x in train.columns if x.endswith('_pred')]
	}
	rfMultiGrid2 = {
		'model': RandomForestClassifier(n_jobs=-1,random_state=SEED),
		'grid': {
			  'n_estimators':[500],
			  'max_depth':[15,20],
			  'min_samples_split':[2],
			  'min_samples_leaf':[2],
			  'max_features':[4,6,8]
		},
		'vars': [x for x in train.columns if x.endswith('ensemble_pred')]
	}
	gbmMultiGrid = {
		'model': GradientBoostingClassifier(random_state=SEED),
		'grid': {
			'n_estimators':[100],
			'learning_rate':[0.025],
			'max_depth':[15],
			'min_samples_split':[2],
			'min_samples_leaf':[2],
			'max_features':[5]
		},    
		'vars': [x for x in train.columns if x.endswith('_pred')]
	}
	nnetRawGrid = {
		'model': NeuralNet(layers=[
					('input', InputLayer),
					('dropoutf', DropoutLayer),
					('dense0', DenseLayer),
					('dropout', DropoutLayer),
					('dense1', DenseLayer),
					('dropout2', DropoutLayer), 
					('output', DenseLayer)
				],
				input_shape=(None,len([v for v in train.columns if not v.endswith('pred')])),
				output_num_units=len(target.unique()),
				output_nonlinearity=softmax,
				update=nesterov_momentum,
				verbose=0),
		'grid': {
			'dropoutf_p':[0.2],
			'dense0_num_units':[1000],
			'dropout_p':[0.3],
			'dense1_num_units':[500],
			'dropout2_p':[0.4],
			'eval_size':[0.01],
			'update_learning_rate': [theano.shared(float32(0.01))],
			'update_momentum': [theano.shared(float32(0.9))],
			'on_epoch_finished': [[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9, stop=0.999)]],
			'max_epochs': [60]
		},    
		'vars': [v for v in train.columns if not v.endswith('pred')]
	}

	trainPred['rf_all'], testPred['rf_all'] = trainMultiModel(MultiModel('RF Multi',rfMultiGrid['model']),target,train[rfMultiGrid['vars']],test[rfMultiGrid['vars']],rfMultiGrid['grid'],cv)
	trainPred['rf_ensemble'], testPred['rf_ensemble'] = trainMultiModel(MultiModel('RF Multi',rfMultiGrid2['model']),target,train[rfMultiGrid2['vars']],test[rfMultiGrid2['vars']],rfMultiGrid2['grid'],cv)
	trainPred['gbm_all'], testPred['gbm_all'] = trainMultiModel(MultiModel('GBM Multi',gbmMultiGrid['model']),target_nnet,train[gbmMultiGrid['vars']],test[gbmMultiGrid['vars']],gbmMultiGrid['grid'],cv)
	trainPred['nnet_raw'], testPred['nnet_raw'] = trainMultiModel(MultiModel('NNET Multi',nnetRawGrid['model']),target_nnet,train[nnetRawGrid['vars']],test[nnetRawGrid['vars']],nnetRawGrid['grid'],cv)
	
	## Ensemble Models ##
	weights = ensembleModels(trainPred,target,15,SEED)
	finalTrainPred = pd.DataFrame(applyWeights(trainPred,weights),columns=target.unique())
	finalTestPred = pd.DataFrame(applyWeights(testPred,weights),columns=target.unique())
	finalTestPred['id'] = id
	finalTestPred = finalTestPred[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]
	print "\tConfirmed Log Loss:",log_loss(target,finalTrainPred.values)

	## Save Results ##
	finalTrainPred.to_csv("../data/train_pred_"+str(k)+"_"+log+"_"+str(pca_n)+"_"+str(rfe)+"_"+str(SEED)+".csv",index=False)
	finalTestPred.to_csv("../data/test_pred_"+str(k)+"_"+log+"_"+str(pca_n)+"_"+str(rfe)+"_"+str(SEED)+".csv",index=False)

if __name__ == "__main__":
	main(path=args.path,k=args.k,log=args.log,pca_n=args.pca_n,rfe=args.rfe,SEED=args.SEED)


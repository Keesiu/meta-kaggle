def loadData(path="../data/",k=5,log='add',pca_n=0,SEED=34):
	from pandas import DataFrame, read_csv
	from numpy import log as ln
	from sklearn.cross_validation import KFold
	from sklearn.preprocessing import LabelEncoder
	from sklearn.preprocessing import StandardScaler
	train = read_csv(path+"train.csv")
	test = read_csv(path+"test.csv")
	id = test.id
	target = train.target
	encoder = LabelEncoder()
	target_nnet = encoder.fit_transform(target).astype('int32')
	feat_names = [x for x in train.columns if x.startswith('feat')]
	train = train[feat_names].astype(float)
	test = test[feat_names]
	if log == 'add':
		for v in train.columns:
			train[v+'_log'] = ln(train[v]+1)
			test[v+'_log'] = ln(test[v]+1)
	elif log == 'replace':
		for v in train.columns:
			train[v] = ln(train[v]+1)
			test[v] = ln(test[v]+1)      
	if pca_n > 0:
		from sklearn.decomposition import PCA
		pca = PCA(pca_n)
		train = pca.fit_transform(train)
		test = pca.transform(test)
	scaler = StandardScaler()
	scaler.fit(train)
	train = DataFrame(scaler.transform(train),columns=['feat_'+str(x) for x in range(train.shape[1])])
	test = DataFrame(scaler.transform(test),columns=['feat_'+str(x) for x in range(train.shape[1])])
	cv = KFold(len(train), n_folds=k, shuffle=True, random_state=SEED)
	return train, test, target, target_nnet, id, cv, encoder

def featSelect(label,trainSet,trainObs,cv,numFeat=5,SEED=34,name=''):
	from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import roc_auc_score
	from numpy import zeros
	model = LogisticRegression(random_state=SEED)
	predCv = zeros(len(trainObs))
	rfe = RFE(model, numFeat, step=1)
	rfe.fit(trainSet,trainObs)
	vars = list(trainSet.columns[rfe.ranking_ == 1])
	auc = 0
	for i in range(1,max(rfe.ranking_)):
		for tr, vl in cv:
			model.fit(trainSet[vars + list(trainSet.columns[rfe.ranking_ == i])].ix[tr],trainObs[tr])
			predCv[vl] = model.predict_proba(trainSet[vars + list(trainSet.columns[rfe.ranking_ == i])].ix[vl])[:,1]
		if roc_auc_score(trainObs,predCv) > auc:
			auc = roc_auc_score(trainObs,predCv)
			vars += list(trainSet.columns[rfe.ranking_ == i])
	for v in vars:
		for tr, vl in cv:
			model.fit(trainSet[[x for x in vars if x != v]].ix[tr],trainObs[tr])
			predCv[vl] = model.predict_proba(trainSet[[x for x in vars if x != v]].ix[vl])[:,1]
		if roc_auc_score(trainObs,predCv) > auc:
			auc = roc_auc_score(trainObs,predCv)
			vars.remove(v)
	for v in [x for x in trainSet.columns if x not in vars]:
		for tr, vl in cv:
			model.fit(trainSet[vars + [v]].ix[tr],trainObs[tr])
			predCv[vl] = model.predict_proba(trainSet[vars + [v]].ix[vl])[:,1]
		if roc_auc_score(trainObs,predCv) > auc:
			auc = roc_auc_score(trainObs,predCv)
			vars += [v]
	print(name,"Final AUC:  ",auc)
	return {label: vars}


def trainModel(label,bestModel,obs,trainSet,testSet,modelgrid,cv,optMetric='auc'):
    """ Train a message classification model """
    from copy import copy
    from numpy import zeros, unique
    from itertools import product
    pred = zeros(len(obs))
    fullpred = zeros((len(obs),len(unique(obs))))
    model = copy(bestModel.model)
    #find the best model via tuning grid
    for tune in [dict(list(zip(modelgrid, v))) for v in product(*list(modelgrid.values()))]:
        for k in list(tune.keys()):
            setattr(model,k,tune[k])
        i = 0
        for tr, vl in cv:
            model.fit(trainSet.ix[tr].values,obs[tr])
            pred[vl] = model.predict_proba(trainSet.ix[vl].values)[:,1]
            fullpred[vl,:] = model.predict_proba(trainSet.ix[vl].values)
            i += 1
        bestModel.updateModel(pred,fullpred,obs,model,trainSet.columns.values,tune,optMetric=optMetric)
    #re-train with all training data
    bestModel.model.fit(trainSet.values,obs)
    print(bestModel)
    return {label: {'pred': pred, 'test_pred':bestModel.model.predict_proba(testSet)[:,1]}}
    

def trainMultiModel(bestModel,obs,trainSet,testSet,modelgrid,cv):
    """ Train a message classification model """
    from copy import copy
    from numpy import zeros, unique
    from itertools import product
    pred = zeros((len(obs),len(unique(obs))))
    model = copy(bestModel.model)
    #find the best model via tuning grid
    for tune in [dict(list(zip(modelgrid, v))) for v in product(*list(modelgrid.values()))]:
        for k in list(tune.keys()):
            setattr(model,k,tune[k])
        i = 0
        for tr, vl in cv:
            model.fit(trainSet.ix[tr].values,obs[tr])
            pred[vl,:] = model.predict_proba(trainSet.ix[vl].values)
            i += 1
        bestModel.updateModel(pred,obs,model,trainSet.columns.values,tune)
    #re-train with all training data
    bestModel.model.fit(trainSet.values,obs)
    print(bestModel)
    return pred, bestModel.model.predict_proba(testSet)

def combineModels(weights,predList,target):
    from sklearn.metrics import log_loss
    from numpy import sum
    weights = weights/sum(weights)
    for weight, pred in zip(weights, predList):
        try:
            result += weight*pred
        except NameError:
            result = weight*pred
    return log_loss(target, result)

def listToDict(l):
	d = {}
	for i in l:
		d[list(i.keys())[0]] = list(i.values())[0]
	return d
	
def applyWeights(predList,weights):
	from numpy import zeros
	pred = zeros(predList[list(predList.keys())[0]].shape)
	for key in predList:
		pred += predList[key]*weights[key]
	return pred
	
def getWeights(multiPred,target,SEED):
	from numpy.random import rand, seed
	from numpy import sum
	from scipy.optimize import minimize
	seed(SEED)
	weights = minimize(combineModels, rand(1,len(multiPred)),args=(list(multiPred.values()),list(target.values)), method='L-BFGS-B', bounds=[(0,1)]*len(multiPred))
	finalWeights = {}
	for i in range(len(multiPred)):
		finalWeights[list(multiPred.keys())[i]] = (weights['x']/sum(weights['x']))[i]
	return weights
	
def ensembleModels(multiPred,target,n=10,SEED=34):
	from numpy import sum
	from scipy.optimize import minimize
	weights = [getWeights(multiPred,target,SEED+i) for i in range(n)]
	finalWeights = [0]*len(multiPred)
	for w in weights:
		finalWeights += (w['x']/sum(w['x']))/10	
	weights = minimize(combineModels, finalWeights,args=(list(multiPred.values()),list(target.values)), method='L-BFGS-B', bounds=[(0,1)]*len(multiPred))
	finalWeights = {}
	for i in range(len(multiPred)):
		finalWeights[list(multiPred.keys())[i]] = (weights['x']/sum(weights['x']))[i]
	print("Results for the Final Ensembled Model.")
	print("\tLog Loss:",weights['fun'])
	print("\tWeights:",finalWeights)
	return finalWeights
	
def float32(k):
	from numpy import cast
	return cast['float32'](k)  


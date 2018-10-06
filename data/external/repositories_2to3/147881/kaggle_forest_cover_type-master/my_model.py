#!/usr/bin/python

import os

import matplotlib
from functools import reduce
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd
import gzip
import pickle as pickle

from sklearn import cross_validation

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.decomposition import PCA, FastICA, KernelPCA, ProbabilisticPCA

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score, log_loss

def gaussian(x, mu, sig):
    return np.exp(-(x-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))

def fit_func(x, *p):
    return p[2] * gaussian(x, p[0], p[1])

def create_html_page_of_plots(list_of_plots):
    if not os.path.exists('html'):
        os.makedirs('html')
    os.system('mv *.png html')
    print(list_of_plots)
    with open('html/index.html', 'w') as htmlfile:
        htmlfile.write('<!DOCTYPE html><html><body><div>')
        for plot in list_of_plots:
            htmlfile.write('<p><img src="%s"></p>' % plot)
        htmlfile.write('</div></html></html>')

def get_plots(in_df):
    list_of_plots = []
    print(in_df.columns)

    for c in in_df.columns:
        if c in ('Id', 'Cover_Type'):
            continue
        pl.clf()
        nent = len(in_df[c])
        hmin, hmax = in_df[c].min(), in_df[c].max()
        xbins = np.linspace(hmin,hmax,nent//500)
        for n in range(1,8):
            covtype = in_df.Cover_Type == n
            a = in_df[covtype][c].values
            #b = in_df[covtype][c].hist(bins=xbins, histtype='step')
            pl.hist(a, bins=xbins, histtype='step')
            #if c == 'Elevation':
                #mu, sig = a.mean(), a.std()
                #x = np.linspace(hmin,hmax,1000)
                #y = (a.sum()/len(xbins)) * gaussian(x, mu, sig)
                #pl.plot(x, y, '--')
        pl.title(c)
        pl.savefig('%s.png' % c)
        list_of_plots.append('%s.png' % c)
    create_html_page_of_plots(list_of_plots)

def plot_failures(in_array, covertype):
    print(in_array.shape)
    list_of_plots = []

    for c in range(in_array.shape[1]):
        pl.clf()
        nent = in_array.shape[0]
        hmin, hmax = in_array[:,c].min(), in_array[:,c].max()
        xbins = np.linspace(hmin,hmax,20)
        for n in range(1,8):
            covtype = covertype == n
            a = in_array[covtype][:,c]
            pl.hist(a, bins=xbins, histtype='step')
        pl.title(c)
        pl.savefig('%s.png' % c)
        list_of_plots.append('%s.png' % c)
    create_html_page_of_plots(list_of_plots)


def transform_from_classes(inp):
    y = np.zeros((inp.shape[0], 7), dtype=np.int64)
    for (index, Class) in enumerate(inp):
        cidx = Class-1
        y[index, cidx] = 1.0
    return y

def transform_to_class(yinp):
    return np.array([x+1 for x in np.argmax(yinp, axis=1)])


def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    ssub_df = pd.read_csv('sampleSubmission.csv')

    #get_plots(train_df)

    labels_to_drop = []

    xtrain = train_df.drop(labels=['Id','Cover_Type']+labels_to_drop, axis=1).values
    ytrain = transform_from_classes(train_df['Cover_Type'].values)
    #ytrain = train_df['Cover_Type'].values
    xtest = test_df.drop(labels=['Id']+labels_to_drop, axis=1).values
    ytest = ssub_df['Id'].values

    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

    return xtrain, ytrain, xtest, ytest

def scorer(estimator, X, y):
    ypred = estimator.predict(X)
    return accuracy_score(ypred, y)

def train_model_parallel(model, xtrain, ytrain, index):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    #xTrain, xTest, yTrain, yTest = \
      #cross_validation.train_test_split(xtrain, ytrain[:,index], test_size=0.4,
                                        #random_state=randint)
    xTrain, yTrain = xtrain, ytrain[:,index]
    #n_est = [10, 100, 200]
    #m_dep = [5, 10, 40]

    #model = GridSearchCV(estimator=model,
                                #param_grid=dict(n_estimators=n_est, max_depth=m_dep),
                                #scoring=scorer,
                                #n_jobs=-1, verbose=1)
    model.fit(xTrain, yTrain)
    print(model)

    #ytest_pred = model.predict(xTest)
    #ytest_prob = model.predict_proba(xTest)
    #print 'accuracy', accuracy_score(ytest_pred,yTest)
    #print 'logloss', log_loss(yTest, ytest_prob)
    with gzip.open('model_%d.pkl.gz' % index, 'wb') as mfile:
        pickle.dump(model, mfile, protocol=2)
    return

def test_model_parallel(xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = \
      cross_validation.train_test_split(xtrain, ytrain, test_size=0.4,
                                        random_state=randint)
    ytest_prob = np.zeros((yTest.shape[0], 7, 2))
    for n in range(7):
        with gzip.open('model_%d.pkl.gz' % n, 'rb') as mfile:
            model = pickle.load(mfile)
            #print 'grid scores', model.grid_scores_
            #print 'best score', model.best_score_
            #print 'best params', model.best_params_
            ytest_prob[:,n,:] = model.predict_proba(xTest)
    #print accuracy_score
    ytest = transform_to_class(yTest).astype(np.int64)
    ytest_pred = transform_to_class(ytest_prob[:,:,1]).astype(np.int64)
    print(ytest.shape, ytest_pred.shape)
    print(accuracy_score(ytest, ytest_pred))

def prepare_submission_parallel(xtrain, ytrain, xtest, ytest):
    print(ytest.shape)
    ytest_prob = np.zeros((ytest.shape[0], 7, 2))
    for n in range(7):
        with gzip.open('model_%d.pkl.gz' % n, 'rb') as mfile:
            model = pickle.load(mfile)
            ytest_prob[:,n,:] = model.predict_proba(xtest)
    ytest2 = transform_to_class(ytest_prob[:,:,1]).astype(np.int64)
    
    df = pd.DataFrame({'Id': ytest, 'Cover_Type': ytest2}, columns=('Id', 'Cover_Type'))
    df.to_csv('submission.csv', index=False)

    return

#def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    #model.fit(xtrain, ytrain)
    #ytest2 = transform_to_class(model.predict(xtest).astype(np.int64))
    ##dateobj = map(datetime.datetime.fromtimestamp, ytest)

    #df = pd.DataFrame({'Id': ytest, 'Cover_Type': ytest2}, columns=('Id', 'Cover_Type'))
    #df.to_csv('submission.csv', index=False)

    #return


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    #model = RandomForestRegressor(n_jobs=-1)
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    #model = DecisionTreeClassifier()
    #model = GradientBoostingClassifier(loss='deviance', verbose=1)

    index = -1
    for arg in os.sys.argv:
        try:
            index = int(arg)
            break
        except ValueError:
            continue
    if index == -1:
        for idx in range(7):
            train_model_parallel(model, xtrain, ytrain, idx)
        prepare_submission_parallel(xtrain, ytrain, xtest, ytest)
    elif index >= 0 and index < 7:
        train_model_parallel(model, xtrain, ytrain, index)
    elif index == 7:
        test_model_parallel(xtrain, ytrain)
    elif index == 8:
        prepare_submission_parallel(xtrain, ytrain, xtest, ytest)

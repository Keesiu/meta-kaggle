#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

import datetime

from pandas.tools.plotting import scatter_matrix

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import cross_val_score

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

def get_plots(in_df, prefix=''):
    list_of_plots = []
    print in_df.columns

    pl.clf()
    scatter_matrix(in_df)
    pl.savefig('%s_matrix.png' % prefix)
    list_of_plots.append('%s_matrix.png' % prefix)

    for c in in_df.columns:
        #if c in ('Id', 'Cover_Type'):
            #continue
        pl.clf()
        nent = len(in_df[c])
        hmin, hmax = in_df[c].min(), in_df[c].max()
        #xbins = np.linspace(hmin,hmax,nent//500)
        pl.hist(in_df[c].values, histtype='step')
        #for n in range(1,8):
            #covtype = in_df.Cover_Type == n
            #a = in_df[covtype][c].values
            ##b = in_df[covtype][c].hist(bins=xbins, histtype='step')
            #pl.hist(a, bins=xbins, histtype='step')
            ##if c == 'Elevation':
                ##mu, sig = a.mean(), a.std()
                ##x = np.linspace(hmin,hmax,1000)
                ##y = (a.sum()/len(xbins)) * gaussian(x, mu, sig)
                ##pl.plot(x, y, '--')
        pl.title('%s %s' % (prefix, c))
        pl.savefig('%s_%s.png' % (prefix,c))
        list_of_plots.append('%s_%s.png' % (prefix,c))
    #create_html_page_of_plots(list_of_plots)
    return list_of_plots


def load_data():
    train_df = pd.read_csv('train.csv', parse_dates=[0,])
    test_df = pd.read_csv('test.csv', parse_dates=[0,])
    sub_df = pd.read_csv('sampleSubmission.csv', parse_dates=[0,])

    print train_df.columns
    print test_df.columns

    #train_df['datetime'] = train_df['datetime'].map(lambda d: d.strftime("%s")).astype(np.int64)
    #test_df['datetime'] = test_df['datetime'].map(lambda d: d.strftime("%s")).astype(np.int64)
    train_df['hour'] = train_df['datetime'].map(lambda d: d.hour).astype(np.int64)
    test_df['hour'] = test_df['datetime'].map(lambda d: d.hour).astype(np.int64)
    train_df['weekday'] = train_df['datetime'].map(lambda d: d.weekday()).astype(np.int64)
    test_df['weekday'] = test_df['datetime'].map(lambda d: d.weekday()).astype(np.int64)
    train_df = train_df.drop(labels=['datetime'], axis=1)
    test_df = test_df.drop(labels=['datetime'], axis=1)

    #create_html_page_of_plots(get_plots(train_df,
                                        #prefix='train') + \
                                        #get_plots(test_df, prefix='test'))

    print train_df.describe()
    print train_df.columns[:9]

    print test_df.describe()

    for c in train_df.columns:
        print train_df[c].dtype, c, list(train_df.columns).index(c)

    #ytrain = train_df.loc[:,['casual', 'registered', 'count']].values
    ytrain = np.log1p(train_df.loc[:,'count'].values)
    xtrain = train_df.drop(labels=['casual', 'registered', 'count'], axis=1).values
    xtest = test_df.values
    ytest = sub_df['datetime'].values

    return xtrain, ytrain, xtest, ytest

def calculate_rmsle(ypred, yactual):
    N = len(yactual)
    lsum = 0.0
    for i in range(N):
        x, y = ypred[i]+1.0, yactual[i]+1.0
        if x < 1:
            x = 1
        lsum += (np.log(x)-np.log(y))**2
    lsum /= N

    return np.sqrt(lsum)

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    #cvAccuracy = np.mean(cross_val_score(model, xtrain, ytrain, cv=2))
    ytest_pred = model.predict(xTest)
    print 'rmsle', calculate_rmsle(np.exp(ytest_pred)-1, np.exp(yTest)-1)
    return model.score(xTest, yTest)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = (np.exp(model.predict(xtest))-1).astype(np.int64)
    #ytest2 = model.predict(xtest).astype(np.int64)
    #dateobj = map(datetime.datetime.fromtimestamp, ytest)

    df = pd.DataFrame({'datetime': ytest, 'count': ytest2}, columns=('datetime','count'))
    df.to_csv('submission.csv', index=False)

    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

    #pca = KernelPCA(kernel='rbf')
    #x_pca = np.vstack([xtrain, xtest])
    #print x_pca.shape
    #pca.fit(xtrain)

    #xtrain = pca.transform(xtrain)
    #xtest = pca.transform(xtest)

    #compare_models(xtrain, ytrain)
    #model = RandomForestRegressor(n_estimators=400, n_jobs=-1)
    model = GradientBoostingRegressor(loss='ls', verbose=1, max_depth=7, n_estimators=200)

    #print 'score', score_model(model, xtrain, ytrain)
    #print model.feature_importances_

    prepare_submission(model, xtrain, ytrain, xtest, ytest)

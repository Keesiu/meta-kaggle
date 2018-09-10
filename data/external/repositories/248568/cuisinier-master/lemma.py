# -*- coding: utf-8 -*-

"""
Not compatible with Python 3
Created on Fri Dec  4 18:11:16 2015

@author: Sravani Kamisetty
"""

from pandas import Series, DataFrame


from sklearn import tree
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression

def computeAccuracy(Y,YPredicted):
    count = 0
    for i in range(0,len(Y)):
        if Y[i] == YPredicted[i]:
            count = count + 1

    return (count/len(Y) * 100)

# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("resources/train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]


testdf = pd.read_json("resources/test.json")
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]



corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts

# LR, SCV
classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}
clf = LinearSVC()
clf = LogisticRegression()
classifier = grid_search.GridSearchCV(clf, parameters)
classifier=classifier.fit(predictors_tr,targets_tr)

#decision trees
#clf = tree.DecisionTreeClassifier()
#parameters = {'max_depth':[100]}
#classifier=clf.fit(predictors_tr,targets_tr)

predictions_train = classifier.predict(predictors_tr)

predictions=classifier.predict(predictors_ts)
for i in range(0,predictions.size):
    predictions[i] = str(predictions[i])
for i in range(0,predictions_train.size):
    predictions_train[i] = str(predictions_train[i])

#print predictions_train
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

#print computeAccuracy(predictors_tr,predictions_train)
#print predictions_train
#print computeAccuracy(predictions,targets_ts)

#print testdf

testdf[['id' , 'cuisine' ]].to_csv("subTree.csv");

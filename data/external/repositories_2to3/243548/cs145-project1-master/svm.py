#!/usr/bin/env python
#Read trainning data#
import json
from pprint import pprint
from sklearn.svm import SVC
from sklearn import svm
with open('train.json') as data_file:    
    data = json.load(data_file)


def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
dictionary = [];
def featurize(data):
    for d in data:
        feat = []
        ings = d["ingredients"]
        for ing in ings:
            for word in ing.strip().split(' '):
                feat.append(word)
        d["feat"] = feat
all_gredients = [];

featurize(data);
n = len(data);
for i in range(len(data)):
	for j in range(len(data[i]["feat"])):
		all_gredients.append(data[i]["feat"][j])

dictionary = getUniqueWords(all_gredients);



unique = len(dictionary);


train_feature = [[0 for x in range(unique)] for x in range(n)] 

for i in range(len(data)):
	for j in range(len(data[i]["feat"])):
		for k in range(unique):
			if(data[i]["feat"][j] == dictionary[k]):
				train_feature[i][k] += 1
				break

train_label = []

for i in range(n):
	train_label.append(data[i]["cuisine"]) 

#read test data

with open('test.json') as data_test_file:    
    data_test = json.load(data_test_file)
    n_test = len(data_test);
featurize(data_test)
test_feature = [[0 for x in range(unique)] for x in range(n_test)] 

for i in range(n_test):
	for j in range(len(data_test[i]["feat"])):
		for k in range(unique):
			if(data_test[i]["feat"][j] == dictionary[k]):
				test_feature[i][k] +=1
				break

				
#using svm function build svm model#

clf = svm.SVC(C=1.0, kernel='linear', 
degree =1,shrinking=True, probability=False, tol=0.1, cache_size=200, 
class_weight=None, verbose=True, max_iter=10000)
clf.fit(train_feature, train_label) 
pred = clf.predict(test_feature)
#the accuracy is 0.76539, It takes more time than built in linear svm model

#using svm function build svm model, choose poly kernal#
clf = svm.SVC(C=1.0, kernel='poly', 
degree =3,shrinking=True, probability=False, tol=0.001, cache_size=200, 
class_weight=None, verbose=True, max_iter=1000)
clf.fit(train_feature, train_label) 
pred = clf.predict(test_feature)

#linear svc
lin_clf = svm.LinearSVC()
lin_clf.fit(train_feature, train_label)
pred = lin_clf.predict(test_feature)
# the accuracy is 0.77876
#using max_itration = 10000
# the accracy is 0.77886
# C =2 the accuracy is 0.77373
# C = 0.5 the accuracy is 0.78178
# C = 0.1 max = 100000 the accuracy is 0.78550
output = [['id','cuisine']]
for i in range(n_test):
	output.append([data_test[i]["id"],pred[i]]) 

import csv
b = open('test.csv', 'w')
a = csv.writer(b)
a.writerows(output)
b.close()
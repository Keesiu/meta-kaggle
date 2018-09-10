#!/cs145/
#Read trainning data#
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score


import json
with open('train.json') as data_file:    
    data = json.load(data_file)
    
with open('test.json') as data_test_file:    
    data_test = json.load(data_test_file)
   
   

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
#read test data

n_test = len(data_test);
featurize(data_test)
test_feature = [[0 for x in range(unique)] for x in range(n_test)] 

for i in range(n_test):
	for j in range(len(data_test[i]["feat"])):
		for k in range(unique):
			if(data_test[i]["feat"][j] == dictionary[k]):
				test_feature[i][k] +=1
				break


#Transform matrix into tf-idf version
transformer = TfidfTransformer();
train = transformer.fit_transform(train_feature);
test=transformer.fit_transform(test_feature);


target= []

for i in range(n):
	target.append(data[i]["cuisine"]) 
 
 
#Use crossvalidation to determine value of paramter
#10-fold cross-validation with alpha=1 for Multinomial Naive Bayes.
clf3=MultinomialNB(alpha=1, fit_prior=False)
scores=cross_val_score(clf3,train_feature1, target,cv=10,scoring='accuracy')
print scores
#Use average accuracy as an estimate of out-of-sample accuracy
print scores.mean()
#search for an optimal value of alpha of Multinomial Naive Bayes
k_range=range(0,11)
k_scores=[]
for k in k_range:
    a=0.1*k
    clf3=MultinomialNB(alpha=a,fit_prior=False)
    scores=cross_val_score(clf3,train_feature1,target,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores

import matplotlib.pyplot as plt
%matplotlib inline

#plot the value of alpha for Multinomial NB (x-axis) versus the cross-validated accuracy(y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of 100xalpha for Multinomial NB')
plt.ylabel('cross-validated Accuracy')

#predict test data
clfb = MultinomialNB(alpha=0.5, fit_prior=False)
clfb.fit(train, target)
pred=clfb.predict(test)

output = [['id','cuisine']]
for i in range(n_test):
	output.append([data_test[i]["id"],pred[i]]) 


import csv
b = open('test5.csv', 'w')
a = csv.writer(b)
a.writerows(output)
b.close()

====================================================================
import json
with open('train.json') as data_file:    
    data1 = json.load(data_file)
    
with open('test.json') as data_test_file:    
    data_test1 = json.load(data_test_file)





#Instead fo spliting ingredients into words , take it as a whole item
def featurize1(data):
    for d in data:
        feat = []
        ings = d["ingredients"]
        for ing in ings:
                feat.append(ing)
        d["feat"] = feat
all_gredients1 = [];

featurize1(data1);
n1 = len(data1);
for i in range(len(data1)):
	for j in range(len(data1[i]["feat"])):
		all_gredients1.append(data1[i]["feat"][j])

dictionary1 = getUniqueWords(all_gredients1);



unique1 = len(dictionary1);
#Build binary entry matrix. "True" means occurence of the feature
train_feature1 =  [[False for x in range(unique1)] for x in range(n1)] 

for i in range(len(data1)):
	for j in range(len(data1[i]["feat"])):
		for k in range(unique1):
			if(data1[i]["feat"][j] == dictionary1[k]):
				train_feature1[i][k] =True
				break


n_test1 = len(data_test1);
featurize(data_test1)


test_feature1 = [[False for x in range(unique1)] for x in range(n_test1)] 

for i in range(n_test1):
	for j in range(len(data_test1[i]["feat"])):
		for k in range(unique1):
			if(data_test1[i]["feat"][j] == dictionary1[k]):
				test_feature1[i][k] =True
				break

#Use crossvalidation to determine value of paramter
#10-fold cross-validation with alpha=1 for Bernoulli Naive Bayes.
clf3=BernoulliNB(alpha=1, fit_prior=False)
scores=cross_val_score(clf3,train_feature1, target,cv=10,scoring='accuracy')
print scores
#Use average accuracy as an estimate of out-of-sample accuracy
print scores.mean()
#search for an optimal value of alpha of Bernoulli Naive Bayes
k_range=range(0,11)
k_scores=[]
for k in k_range:
    a=0.1*k
    clf3=BernoulliNB(alpha=a,fit_prior=False)
    scores=cross_val_score(clf3,train_feature1,target,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores

import matplotlib.pyplot as plt
%matplotlib inline

#plot the value of alpha for Bernoulli NB (x-axis) versus the cross-validated accuracy(y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of 100xalpha for Bernoulli NB')
plt.ylabel('cross-validated Accuracy')


clfb = BernoulliNB(alpha=0.15, fit_prior=False)
clfb.fit(train_feature1, target)
pred=clfb.predict(test_feature1)

output = [['id','cuisine']]
for i in range(n_test):
	output.append([data_test[i]["id"],pred[i]]) 


import csv
b = open('test4.csv', 'w')
a = csv.writer(b)
a.writerows(output)
b.close()

#!/usr/bin/env python
#Read trainning data#
import json
import time
start = time.time()
with open('train.json') as data_file:    
    data = json.load(data_file)

##### PARAMETERS ######
maxdepth = 500
splitIng = True


def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
dictionary = [];
def featurize(data, splitIng):
    if splitIng:
        for d in data:
            feat = []
            ings = d["ingredients"]
            for ing in ings:
                for word in ing.strip().split(' '):
                    feat.append(word)
            d["feat"] = feat
    else:
        for d in data:
            feat = []
            ings = d["ingredients"]
            for ing in ings:
                   feat.append(ing)
            d["feat"] = feat        
all_gredients = [];

cuisine_data = []

featurize(data, splitIng);
n = len(data);
for i in range(n):
        cuisine_data.append(data[i]["cuisine"])
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

load_time = time.time()
print(str(load_time - start) + ": Finished loading training data")
print(str(len(dictionary)))

#read test data

with open('test.json') as data_test_file:    
    data_test = json.load(data_test_file)
    n_test = len(data_test);
featurize(data_test, splitIng)
test_feature = [[0 for x in range(unique)] for x in range(n_test)] 

for i in range(n_test):
	for j in range(len(data_test[i]["feat"])):
		for k in range(unique):
			if(data_test[i]["feat"][j] == dictionary[k]):
				test_feature[i][k] +=1
				break

######################################################################
######################################################################
from sklearn import tree

maxdepth = None
clf = tree.DecisionTreeClassifier(max_depth=maxdepth)
clf = clf.fit(train_feature, train_label)
pred = clf.predict(test_feature)
# unbroken ing ==> 370s to load training data, 0.61726 accuracy
# broken ing ==> 90.83s to load training, 116.27s to build & predict with tree
#   0.64179 accuracy

print(str(time.time() - load_time) + ": Time to predict with tree")

output = [['id','cuisine']]
for i in range(n_test):
	output.append([data_test[i]["id"],pred[i]]) 

import csv
b = open('test.csv', 'w')
a = csv.writer(b)
a.writerows(output)
b.close()

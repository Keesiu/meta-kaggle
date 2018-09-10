# -*- coding: utf-8 -*-
import json
import csv
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
vec = DictVectorizer()
print "loading training data"
with open('train.json') as data_file:    
    data = json.load(data_file)

clf = tree.DecisionTreeClassifier()

print "get list of unique ingredients and cuisines of current local partition"
cuisines = []
ingredients = []
for recipe in data:
    cuisines.append(recipe["cuisine"])
    for ingredient in recipe["ingredients"]:
        ingredients.append(ingredient)
ingredients = list(set(ingredients))
cuisines = list(set(cuisines))
pprint(cuisines)

print "creating data set and target set"
dataSet = []
targetSet = []
for recipe in data:
    item = {}
    targetItem = {}
    for ing in ingredients:
        item[ing] = 0
    for c in cuisines:
        targetItem[c] = 0
    for ingredient in recipe["ingredients"]:
        item[ingredient] = 1
    targetItem[recipe["cuisine"]] = 1
    dataSet.append(item)
    targetSet.append(targetItem)

print "transform to vector arrays"
dataTrain = vec.fit_transform(dataSet).toarray()
targetTrain = vec.fit_transform(targetSet).toarray()
cuisineArray = vec.get_feature_names()
print "training the decision tree"
clf.fit(dataTrain, targetTrain)

with open('test.json') as testfile:    
    test = json.load(testfile)
testIngredients = []
testIds = []
print "building tests"
for t in test:
    itemIng = {}
    for i in ingredients:
        itemIng[i] = 0
    for ingredient in t["ingredients"]:
        if ingredient in ingredients:
            itemIng[ingredient] = 1
    testIngredients.append(itemIng)
    testIds.append(t["id"])
dataTest = []
for test in testIngredients:
    dataTest.extend(vec.fit_transform(test).toarray())
print "predict test"
#print '{0}, {1}'.format(len(dataTest))
print "length of data test"
print len(dataTest)
resultVec = clf.predict(dataTest)
pprint(resultVec)
print "length of result vector"
print len(resultVec)

print "parse the results"
#what happens when r is all 0
results = []
totalResult = 0
numTimes = 0
for r in resultVec:
    index = 0
    totalResult += 1
    rlen = len(r)
    found = 0
    for c in r:
        if c == 1:
            results.append(cuisineArray[index])
            numTimes += 1
            found = 1
        if rlen == index + 1 and found == 0:
                results.append(cuisineArray[19])
        index += 1
print "length of cuisine results"
print len(results)
print "size of index"
print totalResult
print "number of times results was appended"
print numTimes
print "create final output"     
final = {}
index = 0
for ids in testIds:
    #print '{0}, {1}'.format(ids, index)
    final[ids] = results[index]
    index += 1
print "check length"
print len(set(testIds))
print len(testIds)
print len(set(final))
print len(final)

#writer = csv.writer(open('results2.csv', 'wb'))
#for key, value in final.items():
#   writer.writerow([key, value])

with open('results2.csv', 'wb') as f:
    writer = csv.writer(f)
    for key, value in final.items():
        writer.writerow([key, value])


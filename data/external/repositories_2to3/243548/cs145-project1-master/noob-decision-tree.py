#!/usr/bin/env python
import math
#Read trainning data#
import json
import time
from pprint import pprint
start = time.time()
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

cuisine_data = []

featurize(data);
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

print(str(time.time() - start) + ": Finished loading training data") 
######################################################################
######################################################################

""" take list values e.g. [p, n, k]
        and calculates I(p,n,k)
"""
def information(values):
        result = 0.0
        v_sum = sum(values)
        for v in values:
                if v != 0:
                        result -= float(v)/v_sum*math.log(float(v)/v_sum, 2)
        return result

""" take dictionary attribute in form
        attribute = { "yes":[count for each cuisine], "no":[count for each cuisine]} 
    calculate entropy using attribute
"""
def entropy(attribute):
        result = 0.0
        ing_sum = sum(attribute["yes"])+sum(attribute["no"])
        ing_yes = 0.0
        ing_no = 0.0
        for c in attribute["yes"]:
                ing_yes += c
        for c in attribute["no"]:
                ing_no += c
        result += float(ing_yes)/ing_sum*information(attribute["yes"])
        result += float(ing_no)/ing_sum*information(attribute["no"])
        return result

""" Takes data and searches for tuples that include ingredient 'attribute': ["name", #index]
    Returns list of counts per other ingredients for entry to
    'entropy' function
    Result format:
        {"ing":"ingredient", "yes":[count for each cuisine], "no":[count for each cuisine]}
"""
def prune_attribute(attribute, cuisine_data, feature, label):
        n_label = len(label)
        n_data = len(cuisine_data)
        result = {"ing":attribute[0], "yes":[0 for x in range(n_label)], "no":[0 for x in range(n_label)]}
        for data_index in range(n_data):
                label_index = label.index(cuisine_data[data_index])
                if feature[data_index][attribute[1]] > 0:
                        result["yes"][label_index] += 1
                else:
                        result["no"][label_index] += 1
        return result

def isolated_cuisine(cuisine_data):
        result = data[0]
        for datum in cuisine_data:
                if result != datum:
                        return False
        return True

""" Choose ingredient that maximizes information gain
"""
def choose_ing(cuisine_data, feature, dictionary, label):        
        #calculate info for cuisine choice
        n_label = len(label)
        n_dict = len(dictionary)
        n_data = len(cuisine_data)
        cuisine = [0 for x in range(n_label)]
        for ing_index in range(n_dict):
                for data_index in range(n_data):
                        if feature[data_index][ing_index] > 0:
                                label_index = label.index(cuisine_data[data_index])
                                cuisine[label_index] += 1
        cuisine_info = information(cuisine)

        #for each ing, calculate entropy and info_gain
        info_gain = [0.0 for x in range(n_dict)]
        for ing_index in range(n_dict):
                attribute = [dictionary[ing_index], ing_index]
                info_gain[ing_index] += cuisine_info - entropy(prune_attribute(attribute, cuisine_data, feature, label))

        return dictionary[info_gain.index(max(info_gain))]

def most_common_cuisine(cuisine_data, label):
    n_label = len(label)
    n_data = len(cuisine_data)
    cuisine = [0 for x in range(n_label)]
    for data_index in range(n_data):
        label_index = label.index(cuisine_data[data_index])
        cuisine[label_index] += 1
    cuisine_index = cuisine.index(max(cuisine))
    return label[cuisine_index]

""" Construct decision tree
    Parameter max_depth ensures avoiding endless loop
"""

def construct_tree(cuisine_data, feature, dictionary, label, depth, max_depth):
    tree = []
    if len(cuisine_data) < 10:
        tree.append(cuisine_data[0])
    elif depth < max_depth:
        if len(dictionary) == 1: # last ingredient
            root = dictionary[0]
        else:
            root = choose_ing(cuisine_data, feature, dictionary, label)
        print(str(time.time() - start) + ": Found root - " + root + " at depth " + str(depth))
        tree.append(root)

        #split data into branches
        root_index = dictionary.index(root)
        dictionary.pop(root_index)
        branch0_data = { "data":[],
                         "feature":[] }
        branch1_data = { "data":[],
                         "feature":[] }
        n_data = len(cuisine_data)
        print("Root " + root + " has " + str(len(feature)) + " data")
        for index in range(n_data): # split data for branching
            if feature[index][root_index] > 0:
                feature[index].pop(root_index)
                branch1_data["data"].append(cuisine_data[index])
                branch1_data["feature"].append(feature[index])
            else:
                feature[index].pop(root_index)
                branch0_data["data"].append(cuisine_data[index])
                branch0_data["feature"].append(feature[index])
        if len(branch0_data["data"]) == 0:
            if len(branch1_data["data"]) == 0:
                pass
            elif isolated_cuisine(branch1_data["data"]):
                tree.append(most_common_cuisine(branch1_data["data"], label))
            else:
                tree.append(construct_tree(branch1_data["data"], branch1_data["feature"], dictionary, label, depth+1, max_depth))
        elif len(branch1_data["data"]) == 0:
            if isolated_cuisine(branch0_data["data"]):
                tree.append(most_common_cuisine(branch0_data["data"], label))
            else:
                tree.append(construct_tree(branch0_data["data"], branch0_data["feature"], dictionary, label, depth+1, max_depth))
        elif len(dictionary) == 0: # last ingredient
            tree.append(most_common_cuisine(branch0_data["data"], label))
            tree.append(most_common_cuisine(branch1_data["data"], label))
        else: # 2 non-leaf branches
            if isolated_cuisine(branch1_data["data"]):
                branch1 = branch1_data["data"][0]
            else:
                branch1 = construct_tree(branch1_data["data"], branch1_data["feature"], dictionary, label, depth+1, max_depth)
            if isolated_cuisine(branch0_data["data"]):
                branch0 = branch0_data["data"][0]
            elif len(branch0_data["data"]) < 5000:
                branch0 = most_common_cuisine(branch0_data["data"], label)
            else:
                branch0 = construct_tree(branch0_data["data"], branch0_data["feature"], dictionary, label, depth+1, max_depth)
            tree.append(branch0)
            tree.append(branch1)
    else: # depth == max_depth
        root = most_common_cuisine(cuisine_data, label)
        print(str(time.time() - start) + ": Found ending branch - " + root + " at depth " + str(depth))
        tree.append(root)
    return tree

""" Traverse decision tree using test ingredients
        d-tree in form: [ing, [ing==0 branch], [ing==1 branch]]
        Returns final leaf
"""
def traverse_tree(tree, test):
        if len(tree) == 1:
                return tree[0]
        elif len(tree) == 2:
                return tree[1]
        elif isinstance(tree, str):
                return tree
        elif tree[0] in test:
                return traverse_tree(tree[2], test)
        else:
                return traverse_tree(tree[1], test)
#tree = ["A", ["B", ["Chinese"], ["Indian"]], ["B", ["C", ["American"], ["Korean"]], ["C", ["German"]]]]
#print traverse_tree(tree, ["A", "B", "C"])


print("Constructing tree...")
tree = construct_tree(cuisine_data, train_feature, dictionary, train_label, 1, 50)
print(tree)

end = time.time()
print("Finished at " + str(end - start))

#read test data

with open('test.json') as data_test_file:    
    data_test = json.load(data_test_file)
    n_test = len(data_test);
featurize(data_test)

test_result = []
for i in range(n_test):
        i_id = data_test[i]["id"]
        i_cuisine = traverse_tree(tree, data_test[i]["ingredients"])
        test_result.append({"id":i_id, "cuisine":i_cuisine})
        
output = [['id','cuisine']]
for i in range(n_test):
    output.append([test_result[i]["id"],test_result[i]["cuisine"]])

import csv
b = open('test.csv', 'w')
a = csv.writer(b)
a.writerows(output)
b.close()

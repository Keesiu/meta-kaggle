#!/usr/bin/env python
print("Hello, World!");
import json
from pprint import pprint

with open('train.json') as data_file:    
    data = json.load(data_file)
pprint(data[0]["id"])
pprint(data[0]["ingredients"][0])

def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
dictionary = [];
all_gredients = [];
n = len(data);
for i in range(len(data)):
	for j in range(len(data[i]["ingredients"])):
		all_gredients.append(data[i]["ingredients"][j])

dictionary = getUniqueWords(all_gredients);

pprint(dictionary);

unique = len(dictionary);


feature = [[0 for x in range(unique)] for x in range(n)] 

for i in range(len(data)):
	for j in range(len(data[i]["ingredients"])):
		for k in range(unique):
			if(data[i]["ingredients"][j] == dictionary[k]):
				pprint("match")
				feature[i][k] =1
				break
print((feature[1]))
#for i in len(data):
#	tmp = getUniqueWords(data[i]["ingredients"])
#	dictionary.append(tmp)

#pprint(data)
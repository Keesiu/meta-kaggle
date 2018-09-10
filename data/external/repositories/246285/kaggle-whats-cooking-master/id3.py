import json
import math
from pprint import pprint

# Splits the data set by the ingredient x
def split_data(data, x):
	leftData = []
	rightData = []
	for i in range(len(data)):
		if x in data[i]['ingredients']:
			rightData.append(data[i])
		else:
			leftData.append(data[i])
	return (leftData, rightData)
	
#information
def info(arr):
	NumRecipes = 0.0			
	for i in range(len(arr)):
		NumRecipes = NumRecipes + arr[i]
	sum = 0.0
	for i in range(len(arr)):
		if arr[i] == 0.0:
			continue
		else:
			sum = sum + ( arr[i]/NumRecipes)* (1.0/math.log10(2.0)) * math.log10(arr[i]/NumRecipes)
	return -1.0 * sum

#entropy
def entropy(ingredient, mapOfCuisIngred, mapOfCuis):
	NumRecipes = 0.0
	for x in mapOfCuis:
		NumRecipes = NumRecipes + mapOfCuis[x]
	NumWIngr = 0.0
	for x in mapOfCuisIngred:
		if ingredient in mapOfCuisIngred[x]:
			NumWIngr = NumWIngr + mapOfCuisIngred[x][ingredient]
	
	#create the arrays to pass to info that allows computation of information
	a = []
	b = []

	for x in mapOfCuisIngred:
		if ingredient in mapOfCuisIngred[x]:
			a.append(mapOfCuisIngred[x][ingredient])
	
	for x in mapOfCuisIngred:
		if ingredient in mapOfCuisIngred[x]:
			b.append(mapOfCuis[x] - mapOfCuisIngred[x][ingredient])

	Entr1 = 0
	Entr2 = 0

	Entr1 = (NumWIngr/NumRecipes)*info(a)
	Entr2 = ((NumRecipes - NumWIngr) / NumRecipes)* info(b)

	return Entr1 + Entr2

#creates the decision tree
def build_tree(data):

	#first pass through the data
	cCount, iCount, iList = parse_data(data)
	
	# selects the attribute with the min entropy
	attribute, entropy = attr_select(cCount, iCount, iList) # returns the largest info gain/entropy
	
	# tree is represented with a dictionary of dictionaries
	tree = {}
	tree[0] = attribute
	left, right = split_data(data, attribute)
		
	# Check if the left subtree is a leaf
	a,b,c = parse_data(left)
	if len(a) == 1:
		tree[1] = a.keys()[0]
	else:
		tree[1] = build_tree(left)
	
	# Check if the right subtree is a leaf
	a,b,c = parse_data(right)
	if len(a) == 1:
		tree[2] = a.keys()[0]
	else:
		tree[2] = build_tree(right)
	
	return tree

# selects the ingredient based on entropy
def attr_select(cCount, iCount, iList):
	entropies = {}
	for a in iList:
		entropies[a] = entropy(a, iCount, cCount)
	attr = min(entropies, key=entropies.get)
	
	return (attr,entropies[attr])

#parses the json and returns 2 maps plus the ingredient set
def parse_data(data):
	cCounts = {}
	iCounts = {}
	ingrList = []
	for i in range(len(data)):
		cuisineType = data[i]['cuisine']
		ingredients = data[i]['ingredients']
		# adds the ingredients tot he set
		ingrList = ingrList + ingredients
		
		#increments the cuisine count
		if cCounts.has_key(data[i]['cuisine']):
			cCounts[data[i]['cuisine']] = cCounts[data[i]['cuisine']] + 1
		else:
			cCounts[data[i]['cuisine']] = 1
		
		#increment the ingredient count
		if iCounts.has_key(cuisineType):
			for j in range(len(ingredients)):
				if iCounts[cuisineType].has_key(ingredients[j]):
					iCounts[cuisineType][ingredients[j]] = iCounts[cuisineType][ingredients[j]] + 1
				else:
					iCounts[cuisineType][ingredients[j]] = 1
		#Add a new cuisine
		else:
			iCounts[cuisineType] = {}
			for j in range(len(ingredients)):
				iCounts[cuisineType][ingredients[j]] = 1
				
	return (cCounts, iCounts, set(ingrList))

#traverses the decision tree until it hits a leaf
def decision_tree(dt, ingreds):
	if not isinstance(dt, dict):
		return dt
	else:
		if dt[0] in ingreds:
			return decision_tree(dt[2], ingreds)
		else:
			return decision_tree(dt[1], ingreds)

# initial prune on the data set			
def prune_data(data, iCount, iList):
	totals = {}
	for ingr in iList:
		count = 0
		for cuisine in iCount:
			if iCount[cuisine].has_key(ingr):
				count = count + iCount[cuisine][ingr]
		
		totals[ingr] = count

	#the threshold to prune on
	pruneList = []
	for i in totals:
		if totals[i] < 5:
			pruneList.append(i)

	#removes ingredients
	for recipe in data:
		for ingr in pruneList:
			if ingr in recipe['ingredients']:
				recipe['ingredients'].remove(ingr)

	#removes recipes
	for recipe in data[:]:
		if len(recipe['ingredients']) == 0:
			data.remove(recipe)

#classifies a file
def classify(train, test, out):
	with open(train) as data_file:
		trainData = json.load(data_file)

	with open(test) as data_file:
		testData = json.load(data_file)
		
	#x,y,z = parse_data(trainData)
	#prune_data(trainData, y, z)
	
	dTree = build_tree(trainData)
	
	#write to the file
	f = open(out, "w")
	for i in range(len(testData)):
		id = testData[i]['id']
		ingrs = testData[i]['ingredients']
		cuisine = decision_tree(dTree, ingrs)
		#cuisine = 'italian'
		f.write("%d,%s\n" % (id,cuisine))
	f.close()

def test():
	classify('train20k.json', 'test.json', 'out20kkk.csv')

#with open('trainSnip.json') as data_file:
#    data = json.load(data_file)
# with open('train40k.json') as data_file:
	# data = json.load(data_file)
# x,y,z = parse_data(data)
#prune_data(data, y, z)
# iCount = y
# iList = z
# totals = {}
# for ingr in iList:
	# count = 0
	# for cuisine in iCount:
		# if iCount[cuisine].has_key(ingr):
			# count = count + iCount[cuisine][ingr]
	
	# totals[ingr] = count

# pprint(totals)

#2000 is safe, 300 is eperimental
import sys
sys.setrecursionlimit(3000)
#classify('train40k.json', 'test.json', 'out40ktest.csv')
		
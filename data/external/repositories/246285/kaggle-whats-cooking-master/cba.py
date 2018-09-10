import json
import math
from pprint import pprint
from operator import itemgetter

#parses the json and returns 2 maps plus the ingredient set
def parse_data(data):
	cCounts = {}
	iCounts = {}
	iTotals = {}
	for i in range(len(data)):
		cuisineType = data[i]['cuisine']
		ingredients = data[i]['ingredients']
		
		#increments the cuisine count
		if cCounts.has_key(data[i]['cuisine']):
			cCounts[data[i]['cuisine']] = cCounts[data[i]['cuisine']] + 1
		else:
			cCounts[data[i]['cuisine']] = 1
			
		#increments the ingredient count
		for ingr in ingredients:
			if iTotals.has_key(ingr):
				iTotals[ingr] = iTotals[ingr] + 1
			else:
				iTotals[ingr] = 1
		
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
				
	return (cCounts, iCounts, iTotals)

def calculate_cars(iCounts, iTotals, minSupport, minConfidence, rCount):
	cars = []
	for cuisine in iCounts:
		for ingr in iCounts[cuisine]:
			support = 1.0 * iCounts[cuisine][ingr] / rCount
			confidence = 1.0 * iCounts[cuisine][ingr] / iTotals[ingr]
			if support >= minSupport and confidence >= minConfidence:
				rule = (ingr, cuisine, support, confidence)
				cars.append(rule)
		
	return cars

def create_classifier(cars, data, cCounts):
	classifier = []
	newData = data[:]
	sortedCars = sorted(cars, key=itemgetter(3,2), reverse=True)
	for rule in cars:
		ingr, cuisine, support, confidence = rule
		temp = []
		flag = False
		for recipe in newData:
			if ingr in recipe['ingredients'] and cuisine == recipe['cuisine']:
				temp.append(recipe)
				flag = True
		if flag:		
			for recipe in temp:
				newData.remove(recipe)
			classifier.append((ingr,cuisine))
	
	print "uncovered"
	print len(newData)
	
	default = max(cCounts, key=cCounts.get)
	if len(newData) != 0:
		x,y,z = parse_data(newData)
		default = max(x, key=x.get)
	
	return classifier, default
	
def classify(ingredients, classifier, default):
	for rule in classifier:
		ingr, cuisine = rule
		if ingr in ingredients:
			return cuisine
	return default
			
#classifies a file
def main(train, test, out):
	with open(train) as data_file:
		trainData = json.load(data_file)

	with open(test) as data_file:
		testData = json.load(data_file)
		
	cuisineCounts, ingrCounts, ingrTotals = parse_data(trainData)
	cars = calculate_cars(ingrCounts,ingrTotals,5.0/len(trainData),0.6,len(trainData))
	classifier, default = create_classifier(cars, trainData, cuisineCounts)
	#write to the file
	f = open(out, "w")
	f.write("id,cuisine\n")
	for recipe in testData:
		id = recipe['id']
		ingredients = recipe['ingredients']
		cuisine = classify(ingredients, classifier, default)
		f.write("%d,%s\n" % (id,cuisine))
	f.close()

#everything below this is test code
# with open('train.json') as data_file:
	# trainData = json.load(data_file)

# x,y,z = parse_data(trainData)
#c = calculate_cars(y,z,0,0.5,len(trainData))
#d,e = create_classifier(c, trainData, x)

# def test(sup,con):
	main('train40k.json', 'test.json', 'outcba.csv')
	# c = calculate_cars(y,z,sup,con,len(trainData))
	# d,e = create_classifier(c, trainData, x)

#main('train.json', 'test.json', 'outcbafinal.csv')
		
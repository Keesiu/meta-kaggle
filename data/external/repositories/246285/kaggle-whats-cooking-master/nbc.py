import json
import math
from pprint import pprint

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

def classify(ingredients, rCount, iProbs, cCounts, iSet):
	cuisineScores = {}
	for cuisine in cCounts:
		score = 1.0
		for ingr in iSet:
			if ingr in ingredients:
				score = score * iProbs[ingr][cuisine]
			else:
				score = score * (1.0 - iProbs[ingr][cuisine])
		cuisineScores[cuisine] = score * cCounts[cuisine] / rCount
		
	return max(cuisineScores, key=cuisineScores.get)

def calculate_probabilities(cCounts, iCounts, iSet):
	probs = {}
	for ingr in iSet:
		probs[ingr] = {} 
		for cuisine in cCounts:
			if iCounts[cuisine].has_key(ingr):
				probs[ingr][cuisine] = 1.0 * iCounts[cuisine][ingr] / cCounts[cuisine]
			else:
				probs[ingr][cuisine] = 0.0
	return probs	
			
#classifies a file
def main(train, test, out):
	with open(train) as data_file:
		trainData = json.load(data_file)

	with open(test) as data_file:
		testData = json.load(data_file)
		
	cuisineCounts, ingredCounts, ingredientSet = parse_data(trainData)
	probabilities = calculate_probabilities(cuisineCounts, ingredCounts, ingredientSet)
	
	#write to the file
	f = open(out, "w")
	f.write("id,cuisine\n")
	for recipe in testData:
		id = recipe['id']
		ingredients = recipe['ingredients']
		cuisine = classify(ingredients, len(trainData), probabilities, cuisineCounts, ingredientSet)
		f.write("%d,%s\n" % (id,cuisine))
	f.close()

#everything bleow this is test code
def test():
	main('trainSnip.json', 'test.json', 'outnbc.csv')
	
# with open('trainSnip.json') as data_file:
	# trainData = json.load(data_file)

# x,y,z = parse_data(trainData)
# p = calculate_probabilities(x,y,z)

main('train.json', 'test.json', 'outnbc.csv')
		
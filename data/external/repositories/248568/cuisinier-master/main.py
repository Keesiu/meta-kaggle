# -*- coding: utf-8 -*-

import csv
import json
import logging

from Cuisinier import Recipe, ClassifiedRecipe
from CuisinierTFIDF import CuisinierTFIDF

LOGGING_LEVEL = logging.INFO
TRAINING_FILE = "resources/train.json"
TEST_FILE = "resources/test.json"

# Configure logging
logging.basicConfig(filename="log.txt", filemode="w", level=LOGGING_LEVEL)


def getClassifiedRecipes(file):
    f = open(file)
    recipes = json.loads(f.read())
    f.close()
    return [ClassifiedRecipe(recipe["id"], recipe["cuisine"],
                             recipe["ingredients"])
            for recipe in recipes]


def getRecipes(file):
    f = open(file)
    recipes = json.loads(f.read())
    f.close()
    return [Recipe(recipe["id"], recipe["ingredients"]) for recipe in recipes]


# Writes a list of classified recipes to a given file in CSV format
def writeRecipesCSV(file, recipes):
    print("Writing recipes to \"" + file + "\"")
    with open(file, "w", newline="") as fileToWrite:
        csv_writer = csv.writer(fileToWrite, delimiter=",", quotechar="|",
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["id", "cuisine"])

        for recipe in recipes:
            csv_writer.writerow([recipe.id, recipe.cuisine])


# Tests a Cuisinier against its own training data
def selfTest(cuisinier):
    # Read and parse JSON data
    recipes = getClassifiedRecipes(TRAINING_FILE)
    cuisinier.addRecipes(recipes)

    success = 0
    for recipe in recipes:
        result = cuisinier.classifyRecipe(Recipe(recipe.id,
                                                 recipe.ingredients))
        if result.cuisine == recipe.cuisine:
            success += 1

    print(cuisinier.getAlgorithmType() + " self-test accuracy: " +
          str(success) + "/" + str(len(recipes)) +
          " (" + "{0:.2f}".format(success/len(recipes) * 100) + "%)")


# Run test data and output the results
def generateSubmission(cuisinier):
    # Read and parse JSON data
    trainingRecipes = getClassifiedRecipes(TRAINING_FILE)
    testRecipes = getRecipes(TEST_FILE)

    cuisinierType = cuisinier.getAlgorithmType()
    cuisinier.addRecipes(trainingRecipes)
    print("Classifying " + str(len(testRecipes)) + " recipes with " +
          cuisinierType)
    writeRecipesCSV("results" + cuisinierType + ".csv",
                    cuisinier.classifyRecipes(testRecipes))
    print(cuisinierType + " classification complete")


def test():
    recipesToClassify = getRecipes(TEST_FILE)
    recipes = getClassifiedRecipes(TRAINING_FILE)
    cuisinier = Cuisinier()
    cuisinier.addRecipes(recipes)

    with open('submissionData.csv', 'wb') as fileToWrite:
        csv_writer = csv.writer(fileToWrite)
        csv_writer = csv.writer(fileToWrite, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["id","cuisine"])

        success = 0
        for recipe in recipesToClassify:
            result = cuisinier.classifyRecipe(Recipe(recipe.id,
                                                     recipe.ingredients))
            csv_writer.writerow([recipe.id, result.cuisine])


def main():
    cuisinier = CuisinierTFIDF()
    # Calculate initial accuracy
    selfTest(cuisinier)
    generateSubmission(cuisinier)

main()

# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod
from collections import namedtuple

Recipe = namedtuple("ClassifiedRecipe", "id, ingredients")
ClassifiedRecipe = namedtuple("ClassifiedRecipe", "id, cuisine, ingredients")

"""
Uses ingredients to categorize the cuisine of a recipe.
@author: Alan Kha
"""


class Cuisinier:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.recipes = {}  # <id, Recipe>
        self.cuisineCount = {}  # <cuisine, frequency>
        self.cuisineMatrix = {}  # <cuisine, <ingredient, frequency>>
        self.ingredientCount = {}  # <ingredient, frequency>
        self.ingredientMatrix = {}  # <ingredient, <cuisine, frequency>>
        self.preprocessed = False

    """
    Returns the algorithm identifier.
    @return         Algorithm identifier
    """
    @abstractmethod
    def getAlgorithmType(self):
        return "N/A"

    """
    Adds a ClassifiedRecipe to the knowledge base.
    @param  recipe  ClassifiedRecipe Recipe to be added
    @return         True if successful, false otherwise
    """
    def addRecipe(self, recipe):
        if not isinstance(recipe, ClassifiedRecipe):
            raise TypeError("Cuisinier.addRecipe() takes a ClassifiedRecipe")

        # Add new recipes to knowledge base
        if recipe.id not in self.recipes:
            self.recipes[recipe.id] = recipe

            # Add to ingredient counter
            if recipe.cuisine not in self.cuisineCount:
                self.cuisineCount[recipe.cuisine] = 0
            self.cuisineCount[recipe.cuisine] += 1

            # Initialize cuisine matrix entry
            if recipe.cuisine not in self.cuisineMatrix:
                self.cuisineMatrix[recipe.cuisine] = {}

            # Iterate through ingredients
            for ingredient in recipe.ingredients:
                # Add to cuisine matrix
                if ingredient not in self.cuisineMatrix[recipe.cuisine]:
                    self.cuisineMatrix[recipe.cuisine][ingredient] = 0
                self.cuisineMatrix[recipe.cuisine][ingredient] += 1

                # Add to ingredient counter
                if ingredient not in self.ingredientCount:
                    self.ingredientCount[ingredient] = 0
                self.ingredientCount[ingredient] += 1

                # Add to ingredient matrix
                if ingredient not in self.ingredientMatrix:
                    self.ingredientMatrix[ingredient] = {}
                if recipe.cuisine not in self.ingredientMatrix[ingredient]:
                    self.ingredientMatrix[recipe.cuisine] = 0
                self.ingredientMatrix[recipe.cuisine] += 1

            # Return true if successful
            logging.info("Add recipe " + str(recipe.id) + ":\tSUCCESS")
            self.preprocessed = False
            return True

        logging.info("Add recipe " + str(recipe.id) + ":\tFAIL")
        return False

    """
    Add a list of ClassfiedRecipes.
    @param  recipes List of ClassifedRecipes
    """
    def addRecipes(self, recipes):
        success = 0
        for recipe in recipes:
            if (self.addRecipe(recipe)):
                success += 1

        logging.info(str(success) + "/" + str(len(recipes)) + " recipes added")

    """
    Run any preprocessing necessary before classification after any change to
    the knowledgebase.
    """
    @abstractmethod
    def preprocess(self):
        self.preprocessed = True

    """
    Classifies a given Recipe (called by classifyRecipe() after preprocessing).
    @param  recipe  Recipe to be classified
    @return         ClassfiedRecipe
    """
    @abstractmethod
    def classify(self, recipe):
        # TODO Perform classification
        return ClassifiedRecipe(recipe.id, "unknown", recipe.ingredients)

    """
    Classifies a given Recipe.
    @param  recipe  Recipe to be classified
    @return         ClassfiedRecipe
    """
    def classifyRecipe(self, recipe):
        if not isinstance(recipe, Recipe):
            raise TypeError("Cuisinier.classifyRecipe() takes a Recipe")

        # Run preprocessing if necessary
        if not self.preprocessed:
            self.preprocess()

        return self.classify(recipe)

    """
    Classifies a list of Recipes.
    @param  recipes List of Recipes
    @return         List of ClassfiedRecipes
    """
    def classifyRecipes(self, recipes):
        # Iterate through recipes
        return [self.classifyRecipe(recipe) for recipe in recipes]

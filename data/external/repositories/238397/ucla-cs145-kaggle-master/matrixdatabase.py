"""
matrixdatabase.py

Module to help get us sparse matricies from our cuisine databases
"""

import numpy as np
from scipy.sparse import coo_matrix

class MatrixDatabase(object):

    # Pass in a CuisineDatabase object
    def __init__(self, cuisine_db):
        self._cuisine_db = cuisine_db   # underlying cuisine database
        self._n_ingredients = 0         # count the number of ingredients seen
        self._ingredients = {}          # map ingredient -> its number representation
        self._n_train_recipes = 0       # count the number of training recipes seen
        self._n_test_recipes = 0       # count the number of test recipes seen

        # map ingredients to an index number for the matrix
        for recipe in self._cuisine_db.train_set():
            self._n_train_recipes += 1
            for ingredient in recipe["ingredients"]:
                if not ingredient in self._ingredients:
                    self._ingredients[ingredient] = self._n_ingredients
                    self._n_ingredients += 1
        for recipe in self._cuisine_db.test_set():
            self._n_test_recipes += 1
            for ingredient in recipe["ingredients"]:
                if not ingredient in self._ingredients:
                    self._ingredients[ingredient] = self._n_ingredients
                    self._n_ingredients += 1

    # Returns a tuple: (matrix, classes)
    # matrix: coo_matrix, with dimensions n_train_recipes by n_ingredients
    # classes: vector of length n_train_recipes with the classifications for the matrix
    def make_train_matrix(self):
        # build up the classification vector
        classes = []
        # build up the rows and columns that should populate the sparse matrix
        rows = []
        cols = []
        n_visited = 0
        for recipe in self._cuisine_db.train_set():
            for ingredient in recipe["ingredients"]:
                rows.append(n_visited)
                cols.append(self._ingredients[ingredient])
            classes.append(recipe["cuisine"])
            n_visited += 1
        d = np.ones((len(rows),))
        coom = coo_matrix((d, (rows, cols)), shape=(self._n_train_recipes, self._n_ingredients))
        return (coom, classes)

    # Returns a coo_matrix, with dimensions n_test_recipes by n_ingredients
    def make_test_matrix(self):
        # build up the rows and columns that should populate the sparse matrix
        rows = []
        cols = []
        n_visited = 0
        for recipe in self._cuisine_db.test_set():
            for ingredient in recipe["ingredients"]:
                rows.append(n_visited)
                cols.append(self._ingredients[ingredient])
            classes.append(recipe["cuisine"])
            n_visited += 1
        d = np.ones((len(rows),))
        coom = coo_matrix((d, (rows, cols)), shape=(self._n_test_recipes, self._n_ingredients))
        return coom

    # Returns a coo_matrix, with dimensions 1 by n_ingredients
    def make_row_from_recipe(self, ingredients):
        rows = []
        cols = []
        for ingredient in ingredients:
            rows.append(0)
            cols.append(self._ingredients[ingredient])
        d = np.ones((len(rows),))
        coom = coo_matrix((d, (rows, cols)), shape=(1, self._n_ingredients))
        return coom

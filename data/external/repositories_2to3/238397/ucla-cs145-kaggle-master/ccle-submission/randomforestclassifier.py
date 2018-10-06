"""
randomforestclassifier.py

Builds up a random forest classifier
~71 pct accuracy, 2m43.737 execution with n_estimators=100
"""

import math
from classifier import Classifier
from matrixdatabase import MatrixDatabase
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier as RandForestClassifier

class RandomForestClassifier(Classifier):

    # n_estimators is the number of estimators to use in the random forest
    def __init__(self, n_estimators, matrix_database):
        self._matrix_database = matrix_database     # matrix database
        self._forest = RandForestClassifier(n_estimators = n_estimators)
        self._has_fit = False   # have we fit the data yet?

    def learn(self, ingredients, cuisine):
        # do nothing... we have all the data we need in matrix_database
        return

    # Run the random forest on an unclassified recipe
    def classify(self, ingredients):
        # Fit the training data to create the decision trees
        if not self._has_fit:
            matrix, classes = self._matrix_database.make_train_matrix()
            self._forest = self._forest.fit(matrix, classes)
            print("Fitting completed...")
            self._has_fit = True
        # Run the random forest decision trees
        # We're going a little nonconventional (and slower?) here by running the classifier one at a time
        output = self._forest.predict(self._matrix_database.make_row_from_recipe(ingredients))
        # print "Prediction: " + output[0]
        return output[0]

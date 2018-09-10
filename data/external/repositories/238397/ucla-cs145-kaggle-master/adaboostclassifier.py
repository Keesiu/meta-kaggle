"""
adaboostclassifier.py

Builds up an Ada Boost classifier
~49 pct accuracy, 0m46.098s execution with n_estimators=50
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.ensemble import AdaBoostClassifier as ABC

class AdaBoostClassifier(Classifier):

    def __init__(self, n_estimators, matrix_database):
        self._matrix_database = matrix_database
        self._abc = ABC(n_estimators = n_estimators)
        self._has_fit = False

    def learn(self, ingredients, cuisine):
        # do nothing... we have all the data we need in matrix_database
        return

    def classify(self, ingredients):
        if not self._has_fit:
            matrix, classes = self._matrix_database.make_train_matrix()
            self._abc = self._abc.fit(matrix, classes)
            print "Fitting completed..."
            self._has_fit = True
        output = self._abc.predict(self._matrix_database.make_row_from_recipe(ingredients))
        return output[0]
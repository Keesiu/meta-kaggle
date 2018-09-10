"""
baggingclassifier.py

Builds a bagging classifier
~66 pct accuracy, 1m57.020s execution time with n_estimators=10
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.ensemble import BaggingClassifier as BC

class BaggingClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._bc = BC(n_estimators=10)

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			self._bc = self._bc.fit(matrix, classes)
			print 'Fitting complete...'
			self._has_fit = True
		output = self._bc.predict(self._matrix_database.make_row_from_recipe(ingredients))
		return output[0]
		
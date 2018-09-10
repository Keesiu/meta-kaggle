"""
extratreeclassifier.py

Builds an extra tree classifier
~53 pct accuracy, 0m5.327s execution time
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.tree import ExtraTreeClassifier as ETC

class ExtraTreeClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._etc = ETC()

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			self._etc = self._etc.fit(matrix, classes)
			print 'Fitting complete...'
			self._has_fit = True
		output = self._etc.predict(self._matrix_database.make_row_from_recipe(ingredients))
		return output[0]
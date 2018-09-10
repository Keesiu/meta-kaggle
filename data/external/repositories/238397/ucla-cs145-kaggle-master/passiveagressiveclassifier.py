"""
passiveagressiveclassifier.py

Builds a passive agressive classifier
~79 pct accuracy, 0m8.983s execution time
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.linear_model import PassiveAggressiveClassifier as OCC

class PassiveAgressiveClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._occ = OCC(C=0.0083, n_iter=27, loss='hinge')

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			self._occ = self._occ.fit(matrix, classes)
			print 'Fitting complete...'
			self._has_fit = True
		output = self._occ.predict(self._matrix_database.make_row_from_recipe(ingredients))
		return output[0]
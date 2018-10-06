"""
labelpropagationclassifier.py

Builds a label propagation classifier
Takes a very long time, unknown accuracy/execution time
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.semi_supervised import LabelPropagation

class LabelPropagationClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._lbl = LabelPropagation()

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			matrix = matrix.toarray()
			self._lbl = self._lbl.fit(matrix, classes)
			print('Fitting complete...')
			self._has_fit = True
		output = self._lbl.predict(self._matrix_database.make_row_from_recipe(ingredients).toarray())
		return output[0]
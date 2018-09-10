"""
linearsvclassifier.py

Builds a linear support vector classifier
~78 pct accuracy, 0m10.881s execution time
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.svm import LinearSVC as SVC

class LinearSVClassifier(Classifier):

	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._svc = SVC(C=0.6, tol=1e-5, max_iter=10000, dual=False)

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			self._svc = self._svc.fit(matrix, classes)
			print 'Fitting complete...'
			self._has_fit = True
		output = self._svc.predict(self._matrix_database.make_row_from_recipe(ingredients))
		return output[0]
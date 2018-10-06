"""
probabilitycalibrationclassifier.py

Builds a probability calibration classifier
~78 pct accuracy, 0m55.056s execution time
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.calibration import CalibratedClassifierCV

BASEESTIMATOR=None
METHOD='isotonic'
CV=3

class ProbabilityCalibrationClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._pcc = CalibratedClassifierCV(base_estimator=BASEESTIMATOR, method=METHOD, cv=CV)

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			self._pcc = self._pcc.fit(matrix, classes)
			print('Fitting complete...')
			self._has_fit = True
		output = self._pcc.predict(self._matrix_database.make_row_from_recipe(ingredients))
		return output[0]
"""
randomclassifier.py

Implements a dumb, baseline classifier that randomly picks a class
Accuracy tends to be ~4.8%
"""

from classifier import Classifier
import random

class RandomClassifier(Classifier):

    def __init__(self):
        self._classes = set()

    def learn(self, ingredients, cuisine):
        self._classes.add(cuisine)

    def classify(self, ingredients):
        return random.sample(self._classes, 1)[0]

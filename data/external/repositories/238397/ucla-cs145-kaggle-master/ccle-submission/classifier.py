"""
classifier.py

Contains abstract base class with methods that each classifier must implement
"""

# Abstract class with Classifier methods
class Classifier(object):

    # Teach the classifier about a data entry
    # Input: list of ingredients (strings), cuisine class (string)
    # Returns: nothing
    def learn(self, ingredients, cuisine):
        raise NotImplementedError()

    # Run the classifier on an input data entry
    # Input: list of ingredients (strings)
    # Returns: cuisine class (string)
    def classify(self, ingredients):
        raise NotImplementedError()

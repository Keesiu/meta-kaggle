"""
nbc.py

Implements a Naive Bayes Classifier

As per lecture slides on NBC, assign class by maximizing P(X|C_i) * P(C_i).
"""

from classifier import Classifier

class NaiveBayesClassifier(Classifier):

    def __init__(self):
        self._classes = dict()
        self._ingredients = dict()
        self._total_num_cuisines = 0
        self._total_num_ingredients = 0

    # now we want to find P(ingredient | cuisine)
    # P(cuisine)
    # classify by maximum P(ingredient(s) | cuisine) * P(cuisine)
    # cuisine : {ingredients : dict() , number}
    def learn(self, ingredients, cuisine):
        if cuisine in self._classes:
            self._classes[cuisine]["count"] += 1
        else:
            self._classes[cuisine] = dict()
            self._classes[cuisine]["count"] = 1
            self._classes[cuisine]["ingredients"] = dict()

        self._total_num_cuisines += 1

        for ingredient in ingredients:
            # local ingredient count
            if ingredient in self._classes[cuisine]["ingredients"]:
                self._classes[cuisine]["ingredients"][ingredient] += 1
            else:
                self._classes[cuisine]["ingredients"][ingredient] = 1

            # global ingredient count
            if ingredient in self._ingredients:
                self._ingredients[ingredient] += 1
            else:
                self._ingredients[ingredient] = 1

            self._total_num_ingredients += 1

    def classify(self, ingredients):
        max_probability = 0
        max_probability_class = ""

        for cuisine in self._classes:
            p_class = float(self._classes[cuisine]["count"]) / float(self._total_num_cuisines)

            p_ingredients_given_class = 1.0
            for ingredient in ingredients:
                if ingredient in self._classes[cuisine]["ingredients"]:
                    p_ingredients_given_class *= (
                        float(self._classes[cuisine]["ingredients"][ingredient]) / float(self._total_num_ingredients))
                else:
                    # psuedocount: handles zero-out issue of encountering
                    p_ingredients_given_class *= (1.0 / float(self._total_num_ingredients))

            if p_ingredients_given_class > max_probability:
                max_probability = p_ingredients_given_class
                max_probability_class = cuisine

        return max_probability_class

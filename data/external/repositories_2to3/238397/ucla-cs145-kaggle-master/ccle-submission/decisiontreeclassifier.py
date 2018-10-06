"""
decisiontreeclassifier.py

Implements a decision tree classifier
"""

import math
from classifier import Classifier

# Leaf node in the decision tree
# decision is one of the cuisine selections
class LeafNode(object):
    def __init__(self, decision = None):
        self.decision = decision

# Internal node in the decision tree
# split_criteria is an ingredient, and we follow yes_node or no_node
# depending on whether that ingredient is present in the current recipe
class InternalNode(object):
    def __init__(self, split_criteria = None, yes_node = None, no_node = None):
        self.split_criteria = split_criteria
        self.yes_node = yes_node
        self.no_node = no_node

# This classifier assumes that every data point has been learn()'d before
# calling classify(); on the first attempt to classify(), we will build the tree
# n_splits is the number of candidate ingredients to consider as internal nodes (see construct_tree())
class DecisionTreeClassifier(Classifier):
    def __init__(self, n_splits):
        self._root = None
        self._data = [] # (ingredients list, cuisine) tuples
        self._n_splits = n_splits

    def learn(self, ingredients, cuisine):
        self._data.append((ingredients, cuisine))

    # This uses entropy as splitting criteria
    def pick_partition(self, proj_database, possible_splits):
        # split_results: dict mapping ingredient -> entropy of that split
        split_results = {}
        for split_ingredient in possible_splits:
            # counters for number of yes and number of no if we use this split
            n_yes = 0
            n_no = 0
            # dicts mapping cuisine -> count within this split
            yes = {}
            no = {}
            for (ingredients, cuisine) in proj_database:
                if split_ingredient in ingredients:
                    n_yes += 1
                    if cuisine in yes:
                        yes[cuisine] += 1
                    else:
                        yes[cuisine] = 1
                else:
                    n_no += 1
                    if cuisine in no:
                        no[cuisine] += 1
                    else:
                        no[cuisine] = 1
            entropy_yes = 0.0
            entropy_no = 0.0
            for cuisine in yes:
                x = float(yes[cuisine])/float(n_yes)
                entropy_yes -= x * math.log(x, 2)
            for cuisine in no:
                x = float(no[cuisine])/float(n_no)
                entropy_no -= x * math.log(x, 2)
            entropy = ((float(n_yes)/float(n_no + n_yes)) * entropy_yes) + ((float(n_no)/float(n_no + n_yes)) * entropy_no)
            split_results[split_ingredient] = entropy
            # print "Entropy of splitting at " + split_ingredient + " is " + str(entropy)
        return min(split_results, key = split_results.get)

    # At the top level, just call self.construct_tree()
    # Returns a node (recursive function)
    def construct_tree(self, proj_database = None, possible_splits = None, best_guess_sofar = None, n_candidates = 200):
        if proj_database == None:
            proj_database = self._data
        if possible_splits == None:  # this will be true at the top level of recusion
            # Ideally, we would use every possible ingredient as a candidate for splitting
            # This is very expensive; instead, here we will use the most common X ingredients
            # as splitting candidates throughout the algorithm
            c = {}
            for (ingredients, cuisine) in proj_database:
                for ingredient in ingredients:
                    if ingredient in c:
                        c[ingredient] += 1
                    else:
                        c[ingredient] = 1
            c_sort = sorted(c, key = c.get, reverse = True)
            possible_splits = set()
            for ingredient in c_sort[0:n_candidates]:
                possible_splits.add(ingredient)
        # Pre-fetch the "majority rules" best guess in this projected database in case we need it
        cuisines = {}
        total_count = 0
        for (ingredients, cuisine) in proj_database:
            total_count += 1
            if cuisine in cuisines:
                cuisines[cuisine] += 1
            else:
                cuisines[cuisine] = 1
        best_count = 0
        best_cuisine = ""
        for cuisine in cuisines:
            if cuisines[cuisine] > best_count:
                best_count = cuisines[cuisine]
                best_cuisine = cuisine
        # print "Best ratio is " + best_cuisine + " at " + str(float(best_count)/float(total_count)) + " %"

        # Threshold for when to terminate as a leaf node
        # In class, we used 1.0 as the threshold
        THRESHOLD = 0.95
        # MODIFICATION FROM COURSE ALGORITHM:
        # If the projected database is empty, we'll try using the best guess (majority rules)
        # from the PREVIOUS level of recursion (as opposed to just having an undefined node)
        if len(proj_database) == 0:
            if best_guess_sofar == None:
                best_guess_sofar = "italian"
            print("Leaf node (default) chosen: " + best_guess_sofar)
            return LeafNode(best_guess_sofar)
        if len(possible_splits) == 0 or float(best_count)/float(total_count) >= THRESHOLD:
            print("Leaf node chosen: " + best_cuisine)
            return LeafNode(best_cuisine)
        else:
            # print "Searching for best partition..."
            split_criteria = self.pick_partition(proj_database, possible_splits)
            print("Internal node chosen: " + split_criteria)
            possible_splits.remove(split_criteria)
            no_proj = [x for x in proj_database if not (split_criteria in x[0])]
            yes_proj = [x for x in proj_database if (split_criteria in x[0])]
            no_node = self.construct_tree(no_proj, possible_splits.copy(), best_cuisine)
            yes_node = self.construct_tree(yes_proj, possible_splits.copy(), best_cuisine)
            return InternalNode(split_criteria = split_criteria, yes_node = yes_node, no_node = no_node)

    # Recursively start at the root and apply the decision tree to a recipe
    def predict(self, ingredients):
        current_node = self._root
        while isinstance(current_node, InternalNode):
            if current_node.split_criteria in ingredients:
                current_node = current_node.yes_node
            else:
                current_node = current_node.no_node
        return current_node.decision

    # Build the tree if needed; then call predict()
    def classify(self, ingredients):
        if self._root == None:
            self._root = self.construct_tree(n_candidates = self._n_splits)
        return self.predict(ingredients)

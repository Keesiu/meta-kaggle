"""
cuisinedatabase.py

Module for the actual "database" I/O (interacting with JSON data),
and for writing out classification results
"""

import json
import csv
import random

# Abstract class with Database methods
class Database(object):
    def train(self, i):
        raise NotImplementedError()
    def test(self, i):
        raise NotImplementedError()

# CuisineDatabase class
# e.g. db = CuisineDatabase('train.json', 'test.json'), db.entry(i)["cuisine"]
class CuisineDatabase(Database):
    def __init__(self, train_filename, test_filename):
        # For now, we'll load the entire JSON files into memory...
        self._trainf = train_filename
        self._testf = test_filename
        with open(self._trainf) as train_file:    
            self._train = json.load(train_file)
        with open(self._testf) as test_file:    
            self._test = json.load(test_file)
        # Variables for the number of train and test recipes
        self.n_train_recipes = len(self._train)
        self.n_test_recipes = len(self._test)

    # Returns the entire train data set
    def train_set(self):
        return self._train

    # Returns the entire test data set
    def test_set(self):
        return self._test

    # Returns a training data dict, with keys "id", "cuisine", and "ingredients"
    def train(self, i):
        if i < len(self._train):
            return self._train[i]
        else:
            return None

    # Returns an unclassified test data dict, with keys "id" and "ingredients"
    def test(self, i):
        if i < len(self._test):
            return self._test[i]
        else:
            return None

# Similar to CuisineDatabase, but we can use this for testing purchases
# With probability p, we will choose a database entry to be part of the
# testing set as opposed to the training set; then we can check accuracy
# of a classification afterwards
# p = 0.0 indicates we should not save any data for the test set
class TestDatabase(Database):
    def __init__(self, train_filename, p=0.15):
        self._trainf = train_filename
        self._train = []    # the training data
        self._test = []     # the test data (with known truth)
        data = []
        with open(self._trainf) as train_file:    
            data = json.load(train_file)
        for entry in data:
            if random.random() < p:
                self._test.append(entry)
            else:
                self._train.append(entry)
        # Variables for the number of train and test recipes
        self.n_train_recipes = len(self._train)
        self.n_test_recipes = len(self._test)

    # Returns the entire train data set
    def train_set(self):
        return self._train

    # Returns the entire test data set
    def test_set(self):
        return self._test

    # Returns a training data dict, with keys "id", "cuisine", and "ingredients"
    def train(self, i):
        if i < len(self._train):
            return self._train[i]
        else:
            return None

    # Returns an unclassified test data dict, with keys "id" and "ingredients"
    def test(self, i):
        if i < len(self._test):
            result = {}
            result["id"] = self._test[i]["id"]
            result["ingredients"] = self._test[i]["ingredients"]
            return result
        else:
            return None

    # Cheat and peek at the correct classification of a "test" entry
    # This is expensive for now (O(n)), since we need to find the entry
    # with the given id number
    def peek(self, idnum):
        for entry in self._test:
            if entry["id"] == idnum:
                return entry["cuisine"]
        return None

    # Check the accuracy of a given classification against our known truth
    # Returns a tuple: (n_correct, n_total)
    def accuracy(self, c):
        n_correct = 0
        n_total = 0
        for entry in self._test:
            if c.classification(entry["id"]) == entry["cuisine"]:
                n_correct += 1
            n_total += 1
        return (n_correct, n_total)

# Classification class
# For the resulting .csv file
# e.g. c = Classification(), c.classify(123, "greek"), c.save_csv('output.csv')
class Classification(object):
    def __init__(self):
        # For now, we'll store the entire classification into memory...
        # Can change this to appending as we go later, if we want
        self._class = {} # map id (int) -> cuisine (string)

    # Save a classification
    def save(self, idnum, cuisine):
        self._class[idnum] = cuisine

    # Get a classification that we have already saved
    def classification(self, idnum):
        return self._class.get(idnum)

    # Write all stored classifications to file
    def save_csv(self, filename):
        writer = csv.writer(open(filename, 'wb'))
        writer.writerow(["id", "cuisine"])
        for idnum, cuisine in list(self._class.items()):
            writer.writerow([idnum, cuisine])

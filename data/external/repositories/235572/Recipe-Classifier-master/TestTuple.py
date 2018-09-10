import numpy as np


class TestTuple(object):
    def __init__(self, id, attributes):
        self.class_label = None
        self.id = id
        self.attributes = attributes
        self.class_probabilities = np.zeros((20,), dtype=np.float)

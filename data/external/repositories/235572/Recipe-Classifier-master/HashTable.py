import numpy as np
from Node import Node


class HashTable(object):

    def __init__(self, table_size):
        self.size = table_size
        self.table = [None] * table_size

    def put(self, key, value):
        index_of_hash_table = self.hash_function(key)

        node = Node(key, value)

        temp = self.table[index_of_hash_table]
        if temp is None:
            self.table[index_of_hash_table] = node
        else:
            temp.add_node(node)

    def get(self, key):
        index_of_hash_table = self.hash_function(key)

        if self.table[index_of_hash_table] is None:
            return -1

        # handles collisions with separate chaining
        node = self.table[index_of_hash_table]
        if node.key == key:
            return node.value

        while node.key != key:
            node = node.next
            if node is None:
                return -1

        if node.key == key:
            return node.value

    def hash_function(self, key):

        key_index = 0

        for i in xrange(0, len(key)):
            key_index += ord(key[i])

        key_index *= ord(key[0])
        key_index *= ord(key[len(key) - 1])
        key_index /= ord(key[len(key) / 2])

        key_index = key_index % self.size

        if key_index < 0:
            key_index += self.size

        return key_index

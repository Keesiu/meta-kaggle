class Node(object):

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

    def add_node(self, node):
        if self.next is None:
            self.next = node
            return 1
        return self.next.add_node(node)

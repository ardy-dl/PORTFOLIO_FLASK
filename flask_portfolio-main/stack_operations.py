class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        self.size = 0

    def push(self, data):
        new_node = Node(data)
        if self.top:
            new_node.next = self.top
        self.top = new_node
        self.size += 1

    def pop(self):
        if not self.is_empty():
            popped_node = self.top.data
            self.top = self.top.next
            popped_node.next = None
            self.size -= 1
            return popped_node.data
        else:
            raise IndexError("pop from an empty stack")

    def peek(self):
        if not self.is_empty():
            return self.top.data
        else:
            raise IndexError("peek from an empty stack")

    def is_empty(self):
        return self.size == 0

    def length(self):
        return self.size

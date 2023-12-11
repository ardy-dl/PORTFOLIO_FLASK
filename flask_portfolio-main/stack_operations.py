class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def pop(self):
        if not self.is_empty():
            popped_item = self.head.data
            self.head = self.head.next
            self.size -= 1
            return popped_item
        else:
            raise IndexError("pop from an empty stack")

    def peek(self):
        if not self.is_empty():
            return self.head.data
        else:
            raise IndexError("peek from an empty stack")

    def is_empty(self):
        return self.size == 0

    def length(self):
        return self.size
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        self.tail = None
        self.size = 0

    def push(self, data):
        new_node = Node(data)
        if self.top:
            new_node.next = self.top
        self.top = new_node
        self.size += 1

    def pop(self):
        if not self.is_empty():
            popped_node = self.top
            self.top = popped_node.next
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
        
    def print(self):
        current_node = self.top
        result = ''

        while current_node is not None:
            result = str(current_node.data) + '' + result
            current_node = current_node.next

        return result

    def is_empty(self):
        return self.size == 0

    def length(self):
        return self.size
    
def infixToPostfix(infix_expression):
    operators = {'+':1, '-':1, '*':2, '/':2, '(':0}
    ops = Stack()
    output = Stack()

    for token in infix_expression:
        if token.isalnum():
            output.push(token)
        elif token == '(':
            ops.push(token)
        elif token == ')':
            while not ops.is_empty() and ops.peek() != '(':
                output.push(ops.pop())
            ops.pop()
        else:
            while not ops.is_empty and operators[token] <= operators[ops.peek()]:
                output.push(ops.pop(token))
            ops.push(token)

    while not ops.is_empty():
        output.push(ops.pop())

    return output.print()


# Example usage:
infix_expression = "(A+B)*(C+D)"
print(infixToPostfix(infix_expression)) # Output: "3524-*+"

    


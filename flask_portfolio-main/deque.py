class Node:
  def __init__(self, data):
    self.data = data
    self.next = None
    self.prev = None

class Deque:
  def __init__(self):
    self.head = None
    self.tail = None
    self._size = 0

  def is_empty(self):
    return self.head is None

  def add_front(self, data):
    new_node = Node(data)
    if self.is_empty():
        self.head = self.tail = new_node
    else:
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node
    self._size += 1

  def add_back(self, data):
    new_node = Node(data)
    if self.is_empty():
        self.head = self.tail = new_node
    else:
        new_node.prev = self.tail
        self.tail.next = new_node
        self.tail = new_node
    self._size += 1

  def remove_front(self):
    if self.is_empty():
      return None
    else:
      temp = self.head
      self.head = self.head.next
      if self.head:
        self.head.prev = None
      else:
        self.tail = None
      self._size -= 1
      return temp.data

  def remove_back(self):
    if self.is_empty():
      return None
    else:
        temp = self.tail.data
        self.tail = self.tail.prev
        if self.tail is not None:
            self.tail.next = None
        else:
            self.head = None
        self._size -= 1
        return temp

  def get_front(self):
    if self.is_empty():
        raise Exception('Deque is empty')
    else:
        print(self.head.data)

  def get_back(self):
    if self.is_empty():
        raise Exception('Deque is empty')
    else:
        print(self.tail.data)
    
  def size(self):
     return self._size
     
#example usage:
d = Deque()
print(d.is_empty())
d.add_back(8)
d.add_back(5)
d.add_front(7)
d.get_front()
d.add_front(10)
print(d.size())
print(d.is_empty())
d.add_front(11)
print(d.remove_back())
print(d.remove_front())
d.add_front(55)
d.get_back()
d.add_back(45)
print(d.size())
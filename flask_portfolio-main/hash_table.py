class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function_1(self, key):
        return key % self.size

    def hash_function_2(self, key):
        return ((1731 * key + 520123) % 524287) % self.size

    def hash_function_3(self, key):
        return hash(key) % self.size

    def set_hash_function(self, choice):
        if choice == 1:
            self.hash_function = self.hash_function_1
        elif choice == 2:
            self.hash_function = self.hash_function_2
        elif choice == 3:
            self.hash_function = self.hash_function_3
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

    def insert(self, key, value):
        index = self.hash_function(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            self.table[index].insert(0, (key, value))

    def delete(self, key):
        index = self.hash_function(key)
        if self.table[index] is not None:
            for i, (k, _) in enumerate(self.table[index]):
                if k == key:
                    del self.table[index][i]
                    break

    def print_table(self):
        for i, slot in enumerate(self.table):
            print(f"{i}: {slot}")

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

    def set_hash_function(self, choice):
        if choice == 1:
            self.hash_function = self.hash_function_1
        elif choice == 2:
            self.hash_function = self.hash_function_2
        elif choice == 3:
            self.hash_function = self.hash_function_3
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")


# Example usage
def process_commands():
    size_of_table = 32
    hash_table = HashTable(size_of_table)

    while True:
        print("Select Hash Function:")
        print("1. Hash Function 1")
        print("2. Hash Function 2")
        print("3. Hash Function 3 (Python Default)")
        print("Enter 'q' to quit")

        choice = input()
        if choice == 'q':
            break

        try:
            choice = int(choice)
            if 1 <= choice <= 3:
                hash_table.set_hash_function(choice)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                continue
        except ValueError:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        num_commands = int(input("Enter the number of commands: "))

        for _ in range(num_commands):
            command = input("Enter command: ")
            if command.startswith("del "):
                key = sum(map(ord, command[4:]))
                hash_table.delete(key)
            else:
                key = sum(map(ord, command))
                hash_table.insert(key, command)

        hash_table.print_table()

if __name__ == "__main__":
    process_commands()

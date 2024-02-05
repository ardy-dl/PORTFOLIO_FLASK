import timeit
import random
from flask import Flask, request, jsonify, render_template
from linear_search import linear_search
from binary_search import binary_search
from exponential_search import exponential_search
from jump_search import jump_search
from interpolation_search import interpolation_search
from ternary_search import ternary_search
from data_set import generate_sorted_list
from stack_operations import infixToPostfix
from queue1 import Queue
from hash_table import HashTable
from trainstation import Graph, shortest_path
from flask import redirect, url_for





app = Flask(__name__)
my_queue = Queue()
deque_elements = []


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')
    
@app.route('/angel')
def angel():
    return render_template('angel.html')
    
@app.route('/denise')
def denise():
    return render_template('denise.html')
    
@app.route('/kat')
def kat():
    return render_template("kat.html")

@app.route('/hernandez')
def hernandez():
    return render_template("hernandez.html")

@app.route('/ronio')
def ronio():
    return render_template("ronio.html")
    
@app.route('/ck')
def ck():
    return render_template('ck.html')

@app.route('/baneka')
def baneka():
    return render_template("baneka.html")

@app.route('/jiro')
def jiro():
    return render_template("jiro.html")

@app.route('/contact')
def contact():
    return render_template("bnkcontact.html")

@app.route('/searchalgo')
def searchalgo():
    return render_template("searchalgo.html")



@app.route('/dequeue_queue')
def dequeue_queue():
    return render_template("dequeue_queue.html")

@app.route('/queue')
def queue():
    return render_template('queue.html')



@app.route("/enqueue/<int:data>")
def enqueue(data):
    my_queue.enqueue(data)
    return jsonify({"elements": my_queue.get_queue_elements()})

@app.route("/dequeue")
def dequeue():
    data = my_queue.dequeue()
    return jsonify({"data": data, "elements": my_queue.get_queue_elements()})

@app.route('/deque')
def deque():
    return render_template("deque.html", deque_elements=deque_elements)

@app.route('/enqueue_front/<int:data>')
def enqueue_front(data):
    deque_elements.insert(0, data)
    return jsonify({"elements": deque_elements})

@app.route('/enqueue_rear/<int:data>')
def enqueue_rear(data):
    deque_elements.append(data)
    return jsonify({"elements": deque_elements})

@app.route('/dequeue_front')
def dequeue_front():
    if deque_elements:
        deque_elements.pop(0)
    return jsonify({"elements": deque_elements})

@app.route('/dequeue_rear')
def dequeue_rear():
    if deque_elements:
        deque_elements.pop()
    return jsonify({"elements": deque_elements})




@app.route("/small_array", methods=["GET", "POST"])
def small_array():



    small_data = generate_sorted_list(100)
    medium_data = generate_sorted_list(1000)
    large_data = generate_sorted_list(10000)
    test_data = ", ".join(map(str, small_data)) # (modify) choose your desired data set in here
    #print(test_data)

    if request.method == "POST":
        array_str = request.form.get("array")
        target_str = request.form.get("target")
        search_type = request.form.get("search_type")

        try:
            array = list(map(int, array_str.split(",")))
            target = int(target_str)
            low, high = 0, len(array) - 1

            result = -1

            if search_type == "exponential":
                execution_time = timeit.timeit("exponential_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = exponential_search(array, target)
            elif search_type == "binary":
                execution_time = timeit.timeit("binary_search(array, target, low, high)", globals={**globals(), "array": array, "target": target, "low": low, "high": high}, number=1) * 1000
                result = binary_search(array, target, low, high)
            elif search_type == "interpolation":
                execution_time = timeit.timeit("interpolation_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = interpolation_search(array, target)
            elif search_type == "jump":
                execution_time = timeit.timeit("jump_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = jump_search(array, target)
            elif search_type == "linear":
                execution_time = timeit.timeit("linear_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = linear_search(array, target)
            elif search_type == "ternary":
                execution_time = timeit.timeit("ternary_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = ternary_search(array, target)

            return render_template("small_array.html", result=result, search_type=search_type, execution_time=execution_time, test_data=test_data)
        
        except ValueError:
            return render_template("small_array.html", error="Invalid input. Ensure the array and target are integers.")
        except IndexError as e:
            return render_template("small_array.html", error=f"IndexError: {str(e)} Please provide a valid target within the range of the array.")
        except Exception as e:
            return render_template("small_array.html", error=f"An unexpected error occurred: {str(e)}")

    return render_template("small_array.html", test_data=test_data)


@app.route("/medium_array", methods=["GET", "POST"])
def medium_array():

    small_data = generate_sorted_list(100)
    medium_data = generate_sorted_list(1000)
    large_data = generate_sorted_list(10000)
    test_data = ", ".join(map(str, medium_data)) # (modify) choose your desired data set in here
    #print(test_data)

    if request.method == "POST":
        array_str = request.form.get("array")
        target_str = request.form.get("target")
        search_type = request.form.get("search_type")

        try:
            array = list(map(int, array_str.split(",")))
            target = int(target_str)
            low, high = 0, len(array) - 1

            result = -1

            if search_type == "exponential":
                execution_time = timeit.timeit("exponential_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = exponential_search(array, target)
            elif search_type == "binary":
                execution_time = timeit.timeit("binary_search(array, target, low, high)", globals={**globals(), "array": array, "target": target, "low": low, "high": high}, number=1) * 1000
                result = binary_search(array, target, low, high)
            elif search_type == "interpolation":
                execution_time = timeit.timeit("interpolation_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = interpolation_search(array, target)
            elif search_type == "jump":
                execution_time = timeit.timeit("jump_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = jump_search(array, target)
            elif search_type == "linear":
                execution_time = timeit.timeit("linear_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = linear_search(array, target)
            elif search_type == "ternary":
                execution_time = timeit.timeit("ternary_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = ternary_search(array, target)

            return render_template("medium_array.html", result=result, search_type=search_type, execution_time=execution_time, test_data=test_data)
        
        except ValueError:
            return render_template("medium_array.html", error="Invalid input. Ensure the array and target are integers.")
        except IndexError as e:
            return render_template("medium_array.html", error=f"IndexError: {str(e)} Please provide a valid target within the range of the array.")
        except Exception as e:
            return render_template("medium_array.html", error=f"An unexpected error occurred: {str(e)}")

    return render_template("medium_array.html", test_data=test_data)

@app.route("/large_array", methods=["GET", "POST"])
def large_array():

    small_data = generate_sorted_list(100)
    medium_data = generate_sorted_list(1000)
    large_data = generate_sorted_list(10000)
    test_data = ", ".join(map(str, large_data)) # (modify) choose your desired data set in here
    #print(test_data)

    if request.method == "POST":
        array_str = request.form.get("array")
        target_str = request.form.get("target")
        search_type = request.form.get("search_type")

        try:
            array = list(map(int, array_str.split(",")))
            target = int(target_str)
            low, high = 0, len(array) - 1

            result = -1

            if search_type == "exponential":
                execution_time = timeit.timeit("exponential_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = exponential_search(array, target)
            elif search_type == "binary":
                execution_time = timeit.timeit("binary_search(array, target, low, high)", globals={**globals(), "array": array, "target": target, "low": low, "high": high}, number=1) * 1000
                result = binary_search(array, target, low, high)
            elif search_type == "interpolation":
                execution_time = timeit.timeit("interpolation_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = interpolation_search(array, target)
            elif search_type == "jump":
                execution_time = timeit.timeit("jump_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = jump_search(array, target)
            elif search_type == "linear":
                execution_time = timeit.timeit("linear_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = linear_search(array, target)
            elif search_type == "ternary":
                execution_time = timeit.timeit("ternary_search(array, target)", globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = ternary_search(array, target)

            return render_template("large_array.html", result=result, search_type=search_type, execution_time=execution_time, test_data=test_data)
        
        except ValueError:
            return render_template("large_array.html", error="Invalid input. Ensure the array and target are integers.")
        except IndexError as e:
            return render_template("large_array.html", error=f"IndexError: {str(e)} Please provide a valid target within the range of the array.")
        except Exception as e:
            return render_template("large_array.html", error=f"An unexpected error occurred: {str(e)}") 

    return render_template("large_array.html", test_data=test_data)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    if not data or "array" not in data or "target" not in data:
        return jsonify({"error": "Invalid request data. Provide 'array' and 'target'."}), 400

    array = data["array"]
    target = data["target"]

    result_iterative = exponential_search(array, target)

    return jsonify({
        "iterative_search_result": result_iterative,
    })

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/convert', methods=['POST'])
def convert():
    infix_expression = request.form['infix_expression']
    output = infixToPostfix(infix_expression)
    return render_template('index.html', output=output)

@app.route('/hash_table')
def hash_table():
    return render_template("hash_table.html")



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
            self.table[index] = [(key, value)]  # Initialize a stack as a list
        else:
            self.table[index].insert(0, (key, value))  # Push at the head of the stack

    def delete(self, key):
        index = self.hash_function(key)
        if self.table[index] is not None:
            for i, (k, _) in enumerate(self.table[index]):
                if k == key:
                    del self.table[index][i]
                    break

    def print_table(self):
        table_str = ""
        for i, slot in enumerate(self.table):
            table_str += f"{i}: {slot}<br>"
        return table_str

    def set_hash_function(self, choice):
        if choice == 1:
            self.hash_function = self.hash_function_1
        elif choice == 2:
            self.hash_function = self.hash_function_2
        elif choice == 3:
            self.hash_function = self.hash_function_3
        else:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

hash_table = HashTable(32)

@app.route('/hash_table', methods=['GET', 'POST'])
def process_commands():
    if request.method == 'POST':
        choice = int(request.form['hash_function'])
        num_commands = int(request.form['num_commands'])
        commands = request.form['commands'].split('\n')

        hash_table.set_hash_function(choice)

        for command in commands:
            if command.startswith("del "):
                key = sum(map(ord, command[4:]))
                hash_table.delete(key)
            else:
                key = sum(map(ord, command))
                hash_table.insert(key, command)

        return render_template('hash_table.html', table=hash_table.print_table())

    return render_template('hash_table.html', table="")


train_network = Graph()

# LRT Line 1 connections
train_network.add_edge("Roosevelt", "Balintawak", 1)
train_network.add_edge("Balintawak", "Monumento", 1)
train_network.add_edge("Monumento", "5th Avenue", 1)
train_network.add_edge("5th Avenue", "R.Papa", 1)
train_network.add_edge("R.Papa", "Abad Santos", 1)
train_network.add_edge("Abad Santos", "Blumentritt", 1)
train_network.add_edge("Blumentritt", "Tayuman", 1)
train_network.add_edge("Tayuman", "Bambang", 1)
train_network.add_edge("Bambang", "Doroteo Jose", 1)
train_network.add_edge("Doroteo Jose", "Carriedo", 1)
train_network.add_edge("Carriedo", "Central Terminal", 1)
train_network.add_edge("Central Terminal", "United Nations", 1)
train_network.add_edge("United Nations", "Pedro Gil", 1)
train_network.add_edge("Pedro Gil", "Quirino", 1)
train_network.add_edge("Quirino", "Vito Cruz", 1)
train_network.add_edge("Vito Cruz", "Gil Puyat", 1)
train_network.add_edge("Gil Puyat", "Libertad", 1)
train_network.add_edge("Libertad", "EDSA", 1)
train_network.add_edge("EDSA", "Baclaran", 1)

# LRT Line 2 connections
train_network.add_edge("Recto", "Legarda", 1)
train_network.add_edge("Legarda", "Pureza", 1)
train_network.add_edge("Pureza", "V.Mapa", 1)
train_network.add_edge("V.Mapa", "J.Ruiz", 1)
train_network.add_edge("J.Ruiz", "Gilmore", 1)
train_network.add_edge("Gilmore", "Betty Go-Belmonte", 1)
train_network.add_edge("Betty Go-Belmonte", "Araneta-Cubao", 1)
train_network.add_edge("Araneta-Cubao", "Anonas", 1)
train_network.add_edge("Anonas", "Katipunan", 1)
train_network.add_edge("Katipunan", "Santolan", 1)
train_network.add_edge("Santolan", "Marikina", 1)
train_network.add_edge("Marikina", "Antipolo", 1)

# MRT Line 3 connections
train_network.add_edge("North Avenue", "Quezon Avenue", 1)
train_network.add_edge("Quezon Avenue", "GMA Kamuning", 1)
train_network.add_edge("GMA Kamuning", "Araneta-Cubao", 1)
train_network.add_edge("Araneta-Cubao", "Santolan-Anapolis", 1)
train_network.add_edge("Santolan-Anapolis", "Ortigas Avenue", 1)
train_network.add_edge("Ortigas Avenue", "Shaw Boulevard", 1)
train_network.add_edge("Shaw Boulevard", "Boni", 1)
train_network.add_edge("Boni", "Guadalupe", 1)
train_network.add_edge("Guadalupe", "Beundia", 1)
train_network.add_edge("Beundia", "Ayala", 1)
train_network.add_edge("Ayala", "Magallanes", 1)
train_network.add_edge("Magallanes", "Taft Avenue", 1)

# Additional connections
train_network.add_edge("Doroteo Jose", "Recto", 1)
train_network.add_edge("EDSA", "Taft Avenue", 1)
train_network.add_edge("Araneta-Cubao", "Araneta-Cubao", 1)


@app.route('/train_locator')
def train_locator():
    stations = get_all_stations()
    return render_template('train_locator.html', stations=stations)

@app.route('/calculate_path', methods=['POST'])
def calculate_path():
    start_station = request.form.get('start_station')
    end_station = request.form.get('end_station')

    shortest_path_taken, num_stations_traveled = shortest_path(train_network.graph, start_station, end_station)

    return render_template('train_locator.html', 
                           stations=get_all_stations(),
                           shortest_path_taken=shortest_path_taken,
                           num_stations_traveled=num_stations_traveled)

def get_all_stations():
    # Extract all unique station names from the graph
    stations = set(station for connected_stations in train_network.graph.values() for station, _ in connected_stations)
    return sorted(stations)



def generate_random_list(number_of_elements):
    return [random.randint(1, 100) for _ in range(number_of_elements)]

def selection_sort(arr):
    for i in range(0, len(arr) - 1):
        minimum_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minimum_index]:
                minimum_index = j
        arr[i], arr[minimum_index] = arr[minimum_index], arr[i]
    return arr


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def merge_sort(arr):
    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr.pop()

    left = []
    right = []

    for element in arr:
        if element < pivot:
            left.append(element)
        else:
            right.append(element)

    return quick_sort(left) + [pivot] + quick_sort(right)


def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


def measure_time(sort_func, arr):
    time_taken = timeit.timeit(lambda: sort_func(arr.copy()), number=1)
    return time_taken * 1000


@app.route('/sorting_algo', methods=['GET', 'POST'])
def sorting_algo():
    number_of_elements = None
    if request.method == 'POST':
        number_of_elements = int(request.form['number_of_elements'])
    return render_template('sorting_algo.html', number_of_elements=number_of_elements)


@app.route('/sort', methods=['POST'])
def sort():
    number_list = [int(num) for num in request.form['input_numbers'].split(',')]
    sort_algorithm = request.form['sort_algorithm']


    sort_function = None
    if sort_algorithm == 'selection_sort':
        sort_function = selection_sort
    elif sort_algorithm == 'bubble_sort':
        sort_function = bubble_sort
    elif sort_algorithm == 'merge_sort':
        sort_function = merge_sort
    elif sort_algorithm == 'quick_sort':
        sort_function = quick_sort
    elif sort_algorithm == 'insertion_sort':
        sort_function = insertion_sort

    if sort_function is None:
        # Handle unrecognized algorithm
        return render_template('error.html', message='Invalid algorithm')

    time_taken = measure_time(sort_function, number_list)
    sorted_numbers = sort_function(number_list.copy())

    return render_template('sorting_algo.html', input_numbers=number_list, sorted_numbers=sorted_numbers, time_taken=time_taken, algorithm=sort_algorithm)


if __name__ == '__main__':
    app.run(debug=True)
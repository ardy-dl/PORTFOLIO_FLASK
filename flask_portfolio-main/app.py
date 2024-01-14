import timeit
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
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/kat')
def kat():
    return render_template("kat.html")

if __name__ == "__main__":
    app.run(debug=True)

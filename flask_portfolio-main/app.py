from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")
    
@app.route('/angel')
def angel():
    return render_template('angel.html')
    
@app.route('/denise')
def denise():
    return render_template('denise.html')
    
@app.route('/kat')
def kat():
    return render_template("kat.html")

if __name__ == "__main__":
    app.run(debug=True)

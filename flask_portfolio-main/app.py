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

@app.route('/hernandez')
def hernandez():
    return render_template("hernandez.html")

@app.route('/ronio')
def ronio():
    return render_template("ronio.html")
    
@app.route('/ck')
def ck():
    return render_template('ck.html')

@app.route('/searchalgo')
def kat():
    return render_template("searchalgo.html")

if __name__ == "__main__":
    app.run(debug=True)

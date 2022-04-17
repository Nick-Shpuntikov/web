from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def test():
    chislo = 0
    return render_template('ecg.html', name=chislo)
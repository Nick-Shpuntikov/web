from flask import Flask, render_template

app = Flask(__name__, template_folder='static')

@app.route("/")
def home():
    return render_template('ecg.html')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
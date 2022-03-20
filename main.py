from flask import Flask, render_template, url_for
import pandas as pd

app = Flask(__name__)



@app.route('/')
@app.route('/home')
def home():
    return render_template('ecg.html')

@app.route('/about_us')
def about_us():
    data = pd.read_csv('ss.csv', sep=",")
    return render_template('about_us.html', tables=[data.to_html()], titles=[''])

@app.route('/products')
def products():
    return render_template('products.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
from flask import Flask, render_template, url_for
import pandas as pd
from main_open import test
app = Flask(__name__)



@app.route('/')
def home():
    result0 = test()[0]
    result1 = test()[1]
    result2 = test()[2]
    result3 = test()[3]
    return render_template('ecg.html', v0=result0, v1=result1, v2=result2, v3=result3)

@app.route('/about_us')
def about_us():
    data = pd.read_csv('ss.csv', sep=",")
    return render_template('about_us.html', tables=[data.to_html()], titles=[''])

@app.route('/products')
def products():
    return render_template('products.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=True)



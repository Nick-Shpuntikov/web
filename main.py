from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('ecg.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/products')
def products():
    return render_template('products.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
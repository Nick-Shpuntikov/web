import os
from flask import Flask, render_template, url_for, request, redirect, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename
from main_open import test
app = Flask(__name__)


UPLOAD_FOLDER = 'ecgs'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods = ['POST', 'GET'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(f.filename)
#         return "File saved successfully"


# @app.route('/uploads/<name>')
# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/', methods = ['POST', 'GET'] )
def home():
    result0 = test()[0]
    result1 = test()[1]
    result2 = test()[2]
    result3 = test()[3]
    data = "Данные не загружены"
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        # сохраняем файл
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv'))
        # return redirect(url_for('download_file', name=filename))
        data = pd.read_csv('ecgs/data.csv', sep=",")
        df_html = data.to_html()
        return render_template('ecg.html', v0=result0, v1=result1, v2=result2, v3=result3) % df_html
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



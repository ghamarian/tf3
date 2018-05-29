import json
from pprint import pprint

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
from flask_bootstrap import Bootstrap
import os
from sklearn.model_selection import train_test_split
from slider_action import SliderSubmit

from werkzeug.utils import secure_filename

from file_upload import DatasetFileForm

WTF_CSRF_SECRET_KEY = 'a random string'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY

config = {}


# @app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/')
def analysis():
    return redirect(url_for('upload'))


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    if request.method == 'POST':
        # return render_template('feature_selection.html', name='Dataset features', data=config['df'])
        return redirect(url_for('feature'))
    return render_template("slider.html")


@app.route('/split', methods=['GET', 'POST'])
def split():
    dataset_file = config['train']
    removed_ext = os.path.splitext(dataset_file)[0]

    train_file = f"{removed_ext}-train.csv"
    test_file = f"{removed_ext}-test.csv"

    percent = int(request.form['percent'])
    dataset = pd.read_csv(dataset_file)
    test_size = (dataset.shape[0] * percent) // 100

    train_df, test_df = train_test_split(dataset, test_size=test_size)

    train_df.to_csv(train_file)
    test_df.to_csv(test_file)

    config['df'] = train_df

    return jsonify({'done': True})


@app.route('/feature')
def feature():
    x = config['df']
    col_number = x.columns.shape[0]
    cat = assign_category(col_number)
    data = (x.iloc[:6, :]).T
    data.reset_index()
    data.insert(0, 'category', cat)

    return render_template("feature_selection.html", name='Dataset features', data=data)


def assign_category(col_numbers):
    cat = np.random.choice(['numerical', 'categorical'], col_numbers)
    return cat


@app.route('/cat_col', methods=['GET', 'POST'])
def cat_col():
    category_list = json.loads(request.form['cat_column'])
    print(category_list, type(category_list))
    return jsonify({'category_list': 2})


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)
            # flash(u"Error in the %s field - %s" % (getattr(form, field).label.text, error))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = DatasetFileForm()
    form.train_file()
    if form.validate_on_submit():
        target = os.path.join(APP_ROOT, "datasets")
        if not os.path.isdir(target):
            os.mkdir(target)

        save_file(target, form.train_file)
        save_file(target, form.test_file)

        if not 'test' in config:
            return redirect(url_for('slider'))
        else:
            return redirect(url_for('feature'))
    flash_errors(form)
    return render_template('upload_file_wtf.html', form=form)


def save_file(target, dataset_form_field):
    dataset_file = dataset_form_field.data
    if dataset_file:
        dataset_filename = secure_filename(dataset_file.filename)
        destination = os.path.join(target, dataset_filename)
        dataset_file.save(destination)
        # TODO it uses the lables of the form field for extracting the name
        config[dataset_form_field.label.text.split()[0].lower()] = destination


if __name__ == '__main__':
    app.run(debug=True)

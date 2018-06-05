import json
from pprint import pprint

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from flask_bootstrap import Bootstrap
import os

from flask_wtf import csrf
from sklearn.model_selection import train_test_split
from flask_wtf.csrf import CSRFError

from feature_selection import FeatureSelection
from forms.parameters_form import GeneralClassifierForm
from forms.submit_form import Submit
import itertools

from werkzeug.utils import secure_filename

from forms.upload_form import DatasetFileForm

SAMPLE_DATA_SIZE = 5

WTF_CSRF_SECRET_KEY = 'a random string'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY
app.config['WTF_CSRF_CHECK_DEFAULTWTF_CSRF_CHECK_DEFAULT'] = False

config = {}


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400


@app.route('/')
def analysis():
    return redirect(url_for('parameters'))


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    form = Submit(id="form")
    if form.validate_on_submit():
        split_train_test(request)
        return redirect(url_for('feature'))
    return render_template("slider.html", form=form)


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(request):
    dataset_file = config['train']
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    test_file = "{}-test.csv".format(removed_ext)
    percent = int(request.form['percent'])
    dataset = pd.read_csv(dataset_file)
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size)
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    config['df'] = train_df


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    # TODO do it once and test this.
    x = config['df']
    x.reset_index(inplace=True, drop=True)
    categories, unique_values = assign_category(x)
    data = (x.head(SAMPLE_DATA_SIZE).T)
    data.insert(0, 'Unique Values', unique_values)
    data.insert(0, 'Category', categories)

    sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
    data.columns = list(itertools.chain(['Category', '#Unique Values'], sample_column_names))

    config['data'] = data

    form = Submit()
    if form.validate_on_submit():
        category_list = json.loads(request.form['cat_column'])
        pprint(category_list)
        config['data'].Category = category_list
        pprint(config['fs'].create_tf_features(category_list))
        return redirect(url_for('target'))

    return render_template("feature_selection.html", name='Dataset features selection', data=config['data'],
                           cat=categories, form=form)

@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    form = GeneralClassifierForm()
    if form.validate_on_submit():
        pprint(request.form)
        return jsonify({'submit': True})
    flash_errors(form)
    return render_template('parameters.html', form=form)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    data = config['data']
    if form.validate_on_submit():
        target = json.loads(request.form['selected_row'])[0]
        pprint(target)
        return redirect(url_for('parameter'))
    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data)


def assign_category(df):
    fs = FeatureSelection(df)
    config['fs'] = fs
    feature_dict = fs.feature_dict()
    unique_vlaues = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    return category_list, unique_vlaues


@app.route('/cat_col', methods=['GET', 'POST'])
def cat_col():
    category_list = json.loads(request.form['cat_column'])
    config['data'].category = category_list
    pprint(config['data'])
    return jsonify({'category_list': 2})


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)
            # flash(u"Error in the %s field - %s" % (getattr(form, field).label.text, error))


# @app.route('/parameters')
# def parameters():
#     form = ParametersForm()
#
#     if form.validate_on_submit():
#         return jsonify({'submit': True})
#
#     return render_template("parameters.html", form=form)

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

import json
from pprint import pprint

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
from flask_bootstrap import Bootstrap
import os

from flask_wtf import csrf
from sklearn.model_selection import train_test_split
from flask_wtf.csrf import CSRFError

from config import config_reader
from config.config_writer import ConfigWriter
from feature_selection import FeatureSelection
from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
from forms.submit_form import Submit
import itertools

from werkzeug.utils import secure_filename

from forms.upload_form import UploadForm
from runner import Runner
DATASETS = "datasets"

SAMPLE_DATA_SIZE = 5

WTF_CSRF_SECRET_KEY = os.urandom(42)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY

config_writer = ConfigWriter()
config = {}


def get_session():
    with app.app_context():
        if 'user' not in session:
            session['user'] = os.urandom(12)
        return session['user']


def get_config():
    user = get_session()
    if user not in config:
        config[user] = {}
    return config[user]


def get(key):
    return get_config()[key]


def update_config(key, value):
    config = get_config()
    config[key] = value


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():

        target = os.path.join(APP_ROOT, DATASETS)
        if not os.path.isdir(target):
            os.mkdir(target)

        if form.is_existing.data:
            train_file_name = form.exisiting_files.data['train_file']
            update_config('train_file', os.path.join(target, train_file_name))
            config_writer.add_item('PATHS', 'train_file', os.path.join(target, train_file_name))

            test_file_name = form.exisiting_files.data['validation_file']
            update_config('validation_file', os.path.join(target, test_file_name))
            config_writer.add_item('PATHS', 'validation_file', os.path.join(target, test_file_name))
        else:
            save_file(target, form.new_files.train_file, 'train_file')
            save_file(target, form.new_files.test_file, 'validation_file')

        if not 'validation_file' in get_config():
            return redirect(url_for('slider'))
        else:
            return redirect(url_for('feature'))
    flash_errors(form)
    return render_template('upload_file_form.html', form=form)


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    form = Submit(id="form")
    if form.validate_on_submit():
        split_train_test(request)
        return redirect(url_for('feature'))
    return render_template("slider.html", form=form)


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    if 'df' not in get_config():
        update_config('df', pd.read_csv(get('train_file')))
    df = get('df')
    df.reset_index(inplace=True, drop=True)
    categories, unique_values, default_list, frequent_values2frequency = assign_category(df)

    data = (df.head(SAMPLE_DATA_SIZE).T)
    data.insert(0, 'Defaults', default_list.values())
    data.insert(0, '(most frequent, frequency)', frequent_values2frequency.values())
    data.insert(0, 'Unique Values', unique_values)
    data.insert(0, 'Category', categories)

    sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
    data.columns = list(
        itertools.chain(['Category', '#Unique Values', '(Most frequent, Frequency)', 'Defaults'], sample_column_names))

    update_config('data', data)

    form = Submit()
    if form.validate_on_submit():
        update_config('category_list', json.loads(request.form['cat_column']))
        default_values = json.loads(request.form['default_column'])
        get('data').Category = get('category_list')
        update_config('defaults', dict(zip(get('data').index.tolist(), default_values)))
        print(get('defaults'))
        return redirect(url_for('target'))

    return render_template("feature_selection.html", name='Dataset features selection',
                           data=get('data'),
                           cat=categories, form=form)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    data = get('data')
    if form.validate_on_submit():
        target = json.loads(request.form['selected_row'])[0]
        update_config('features', get('fs').create_tf_features(get('category_list'), target))
        update_config('target', target)
        return redirect(url_for('parameters'))
    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    target_type = get('data').Category[get('target')]

    if target_type == 'numerical':
        form = GeneralRegressorForm()
        labels = None
    else:
        form = GeneralClassifierForm()
        labels = get('fs').cat_unique_values_dict[get('target')]
    if form.validate_on_submit():
        pprint(request.form)
        config_writer.populate_config(request.form)
        config_writer.write_config('config/new_config.ini')
        CONFIG_FILE = "config/new_config.ini"
        dtypes = get('fs').group_by(get('category_list'))
        all_params_config = config_reader.read_config(CONFIG_FILE)
        runner = Runner(all_params_config, get('features'), get('target'),
                        labels,
                        get('defaults'), dtypes)
        runner.run()
        return jsonify({'submit': True})
    flash_errors(form)
    return render_template('parameters.html', form=form)


@app.route('/')
def main():
    return redirect(url_for('upload'))


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(request):
    dataset_file = get('train_file')
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(request.form['percent'])
    dataset = pd.read_csv(dataset_file)
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)

    update_config('train_file', train_file)
    update_config('validation_file', validation_file)

    config_writer.add_item('PATHS', 'train_file', train_file)
    config_writer.add_item('PATHS', 'validation_file', validation_file)
    update_config('df', train_df)


def assign_category(df):
    fs = FeatureSelection(df)
    update_config('fs', fs)
    feature_dict = fs.feature_dict()
    unique_vlaues = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    default_list = fs.defaults
    frequent_values2frequency = fs.frequent_values2frequency

    return category_list, unique_vlaues, default_list, frequent_values2frequency


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)
            # flash(u"Error in the %s field - %s" % (getattr(form, field).label.text, error))


def save_file(target, dataset_form_field, dataset_type):
    dataset_file = dataset_form_field.data
    if dataset_file:
        dataset_filename = secure_filename(dataset_file.filename)
        destination = os.path.join(target, dataset_filename)
        dataset_file.save(destination)
        update_config(dataset_type, destination)


if __name__ == '__main__':
    app.run(debug=True)

import json
from pprint import pprint

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from flask_bootstrap import Bootstrap
import os

from flask_wtf import csrf
from sklearn.model_selection import train_test_split
from flask_wtf.csrf import CSRFError

from config import config_reader
from config.config_writer import ConfigWriter
from feature_selection import FeatureSelection
from forms.parameters_form import GeneralClassifierForm
from forms.submit_form import Submit
import itertools

from werkzeug.utils import secure_filename

from forms.upload_form import DatasetFileForm
from runner import Runner

SAMPLE_DATA_SIZE = 5

WTF_CSRF_SECRET_KEY = 'a random string'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY

config = {}
config_writer = ConfigWriter()


@app.route('/')
def analysis():
    return redirect(url_for('upload'))


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
    validation_file = "{}-test.csv".format(removed_ext)
    percent = int(request.form['percent'])
    dataset = pd.read_csv(dataset_file)
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size)
    train_df.to_csv(train_file)
    test_df.to_csv(validation_file)

    config_writer.add_item('PATHS', 'training_file', train_file)
    config_writer.add_item('PATHS', 'validation_file', validation_file)
    config['df'] = train_df


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    # TODO do it once and test this.
    x = config['df']
    x.reset_index(inplace=True, drop=True)
    categories, unique_values, default_list, frequent_values2frequency = assign_category(x)

    data = (x.head(SAMPLE_DATA_SIZE).T)
    data.insert(0, 'Defaults', default_list.values())
    data.insert(0, '(most frequent, frequency)', frequent_values2frequency.values())
    data.insert(0, 'Unique Values', unique_values)
    data.insert(0, 'Category', categories)

    sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
    data.columns = list(
        itertools.chain(['Category', '#Unique Values', '(Most frequent, Frequency)', 'Defaults'], sample_column_names))

    config['data'] = data

    form = Submit()
    if form.validate_on_submit():
        config['category_list'] = json.loads(request.form['cat_column'])
        config['default_column'] = json.loads(request.form['default_column'])
        config['data'].Category = config['category_list']
        config['data'].Defaults = config['default_column']
        print(config['data'].Defaults)
        return redirect(url_for('target'))

    return render_template("feature_selection.html", name='Dataset features selection', data=config['data'],
                           cat=categories, form=form)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    form = GeneralClassifierForm()
    if form.validate_on_submit():
        pprint(request.form)
        config_writer.populate_config(request.form)
        config_writer.write_config('config/new_config.ini')
        CONFIG_FILE = "config/new_config.ini"
        all_params_config = config_reader.read_config(CONFIG_FILE)
        runner = Runner(all_params_config, config['features'], config['target'],
                        config['fs'].cat_unique_values_dict[config['target']])
        runner.run()
        return jsonify({'submit': True})
    flash_errors(form)
    return render_template('parameters.html', form=form)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    data = config['data']
    if form.validate_on_submit():
        target = json.loads(request.form['selected_row'])[0]
        config['features'] = config['fs'].create_tf_features(config['category_list'], target)
        config['target'] = target
        # config['fs'].select_target(target)
        # config['features'] = config['fs'].feature_columns
        return redirect(url_for('parameters'))
    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data)


def assign_category(df):
    fs = FeatureSelection(df)
    config['fs'] = fs
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

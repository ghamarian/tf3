import json
import tensorflow as tf
from pprint import pprint
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
from flask_bootstrap import Bootstrap
import os

from sklearn.model_selection import train_test_split

from config import config_reader
from config.config_writer import ConfigWriter
from feature_selection import FeatureSelection
from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
from forms.submit_form import Submit
from forms.run_form import RunForm
import itertools

from werkzeug.utils import secure_filename

from forms.upload_form import UploadForm
from runner import Runner
import logging
import threading
import time
from utils import copyfile
from multiprocessing import Process
import uuid

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

DATASETS = "datasets"

SAMPLE_DATA_SIZE = 5

WTF_CSRF_SECRET_KEY = os.urandom(42)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY

config_writer = ConfigWriter()
config = {}

user_model = {}
stop_event = threading.Event()
processes = {}


def get_session():
    with app.app_context():
        if 'user' not in session:
            session['user'] = uuid.uuid4()
        return session['user']


def get_config():
    user = get_session()
    if user not in config:
        config[user] = {}
        copyfile('config/default_config.ini', 'config/' + str(user) + '.ini')
        config[user]['config_file'] = 'config/' + str(user) + '.ini'
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
            # TODO check if files exists
            if not 'validation_file' in get_config() and not isinstance(form.new_files.train_file.data,
                                                                        str) and not isinstance(
                    form.new_files.test_file.data, str):
                config_writer.add_item('PATHS', 'train_file',
                                       os.path.join(target, form.new_files.train_file.data.filename))
                config_writer.add_item('PATHS', 'validation_file',
                                       os.path.join(target, form.new_files.test_file.data.filename))
                return redirect(url_for('feature'))
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
        update_config('split_df', request.form['percent'])
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
        get('data').Defaults = default_values
        update_config('defaults', dict(zip(get('data').index.tolist(), default_values)))
        get('fs').update(get('category_list'), dict(zip(get('data').index.tolist(), default_values)))
        # categories, unique_values, default_list, frequent_values2frequency = assign_category(get('data'))
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
        if 'split_df' in get_config():
            split_train_test(get('split_df'))
        return redirect(url_for('parameters'))
    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    target_type = get('data').Category[get('target')]
    # CONFIG_FILE = 'config/new_config.ini'
    CONFIG_FILE = config[get_session()]['config_file']
    if target_type == 'numerical':
        form = GeneralRegressorForm()
    else:
        form = GeneralClassifierForm()
    if form.validate_on_submit():
        pprint(request.form)
        config_writer.populate_config(request.form)
        config_writer.write_config(CONFIG_FILE)
        return redirect(url_for('run'))
    flash_errors(form)
    return render_template('parameters.html', form=form)


@app.route('/run', methods=['GET', 'POST'])
def run():
    form = RunForm()
    target_type = get('data').Category[get('target')]
    labels = None if target_type == 'numerical' else get('fs').cat_unique_values_dict[get('target')]
    # CONFIG_FILE = "config/new_config.ini"
    CONFIG_FILE = config[get_session()]['config_file']
    if form.validate_on_submit():
        tboard_thread = threading.Thread(name='tensor_board', target=lambda: tensor_board_thread(CONFIG_FILE))
        tboard_thread.setDaemon(True)
        tboard_thread.start()

        dtypes = get('fs').group_by(get('category_list'))
        all_params_config = config_reader.read_config(CONFIG_FILE)
        r_thread = Process(target=lambda: run_thread(all_params_config, get('features'), get('target'),
                                                     labels, get('defaults'), dtypes), name='run')
        r_thread.daemon = True
        r_thread.start()
        processes[get_session()] = r_thread

        return render_template('run.html', form=form, running=1)
    return render_template('run.html', form=form, running=0)


@app.route('/pause', methods=['GET', 'POST'])
def pause():
    import psutil
    p = processes[get_session()] if get_session() in processes.keys() else None
    if not isinstance(p, str):
        pid = p.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print
            "child", child
            child.kill()
        parent.kill()
        processes[get_session()] = ''
    return redirect(url_for('run'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    features = get('defaults')
    target = get('target')
    dict_types, categoricals = get_dictionaries()
    # CONFIG_FILE = "config/new_config.ini"
    CONFIG_FILE = config[get_session()]['config_file']
    directory = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    checkpoints = get_acc(directory)
    if request.method == 'POST':
        change_model_default(request.form['radiob'], CONFIG_FILE)
        new_features = {}
        for k, v in features.items():
            new_features[k] = request.form[k] if k != target else features[k]

        all_params_config = config_reader.read_config(CONFIG_FILE)
        runner = Runner(all_params_config, get('features'), get('target'),
                        get('fs').cat_unique_values_dict[get('target')],
                        get('defaults'), get('fs').group_by(get('category_list')))

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=input_predict_fn(new_features, dict_types),
                                                              y=None, num_epochs=1, shuffle=False)
        predictions = list(runner.predict(predict_input_fn))
        final_pred = predictions[0]['classes'][0]
        new_features.pop(target)
        return render_template('predict.html', features=new_features, target=get('target'),
                               types=get_html_types(dict_types), categoricals=categoricals,
                               prediction=final_pred.decode("utf-8"), checkpoints=checkpoints)
    sfeatures = features.copy()
    sfeatures.pop(target)
    return render_template('predict.html', features=sfeatures, target=get('target'),
                           types=get_html_types(dict_types), categoricals=categoricals, checkpoints=checkpoints)


@app.route('/')
def main():
    return redirect(url_for('upload'))


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(percent):
    dataset_file = get('train_file')
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(percent)
    dataset = get('df')
    counts = dataset[get('target')].value_counts()
    dataset = dataset[dataset[get('target')].isin(counts[counts > 1].index)]
    target = dataset[[get('target')]]
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=target)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)

    update_config('train_file', train_file)
    update_config('validation_file', validation_file)

    config_writer.add_item('PATHS', 'train_file', train_file)
    config_writer.add_item('PATHS', 'validation_file', validation_file)


def assign_category(df):
    fs = FeatureSelection(df)
    update_config('fs', fs)
    feature_dict = fs.feature_dict()
    unique_vlaues = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    default_list = fs.defaults
    frequent_values2frequency = fs.frequent_values2frequency
    return category_list, unique_vlaues, default_list, frequent_values2frequency


def update_fs(df):
    fs = FeatureSelection(df)
    update_config('fs', fs)


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


def tensor_board_thread(CONFIG_FILE):
    config_path = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    logging.debug('Starting tensor board')
    time.sleep(3)
    os.system("tensorboard --logdir=" + config_path)
    logging.debug('Exiting tensor board')


def run_thread(all_params_config, features, target, labels, defaults, dtypes):
    runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
    runner.run()


def input_predict_fn(features, dict_types):
    input_predict = {}
    for k, v in features.items():
        if dict_types[k] == 'numerical' or dict_types[k] == 'range':
            input_predict[k] = np.array([int(float(v))])
        else:
            input_predict[k] = np.array([v])
    input_predict.pop(get('target'), None)
    return input_predict


def get_html_types(dict_types):
    dict_html_types = {}
    for k, v in dict_types.items():
        dict_html_types[k] = "text" if v == 'categorical' else "number"
    return dict_html_types


def get_dictionaries():
    features = get('defaults')
    dict_types = {}
    categoricals = {}
    cont = 0
    for k, v in features.items():
        dict_types[k] = get('category_list')[cont]
        if get('category_list')[cont] == 'categorical':
            categoricals[k] = get('fs').cat_unique_values_dict[k]
        cont += 1
    categoricals.pop(get('target'))
    return dict_types, categoricals


def change_model_default(new_model, CONFIG_FILE):
    text = 'model_checkpoint_path: "model.ckpt-number"\n'.replace('number', new_model)
    path = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    with open(path + '/checkpoint') as f:
        content = f.readlines()
    content[0] = text
    file = open(path + '/checkpoint', 'w')
    file.write(''.join(content))
    file.close()


def get_acc(directory):
    checkpoints = []
    accuras = {}
    eval_dir = os.path.join(directory, 'eval')
    if os.path.exists(eval_dir):
        files_checkpoints = os.listdir(directory)
        for file in files_checkpoints:
            if '.meta' in file:
                checkpoints.append(file.split('.')[1].split('-')[-1])
        path_to_events_file = os.path.join(eval_dir, os.listdir(eval_dir)[0])
        for e in tf.train.summary_iterator(path_to_events_file):
            if str(e.step) in checkpoints:
                accuras[e.step] = {}
                for v in e.summary.value:
                    if v.tag == 'loss':
                        accuras[e.step]['loss'] = v.simple_value
                    elif v.tag == 'accuracy':
                        accuras[e.step]['accuracy'] = v.simple_value
    return accuras


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

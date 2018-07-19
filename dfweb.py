import itertools
import json
import logging
import os
import pandas as pd
import threading
import time
from feature_selection import FeatureSelection
from utilities import utils_run, utils_custom, utils_io, threads, utils_features, utils_config, upload_util
from config import config_reader
from config.config_writer import ConfigWriter
from db import db
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, login_user, login_required, logout_user
from forms.login_form import LoginForm
from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
from forms.submit_form import Submit
from forms.upload_form import UploadForm, UploadNewForm
from multiprocessing import Process
from pprint import pprint
from sklearn.model_selection import train_test_split
from user import User
from runner import Runner
from werkzeug.security import check_password_hash
from tensorflow.python.platform import gfile
from multiprocessing import Manager
from werkzeug.utils import secure_filename
from shutil import copyfile

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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///username.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

config_writer = ConfigWriter()
config = {}

stop_event = threading.Event()
processes = {}
ports = {}

login_manager = LoginManager()
login_manager.init_app(app)

return_dict = Manager().dict()


def get_session(user_id):
    with app.app_context():
        if user_id not in session:
            return redirect(url_for('login'))
        return session[user_id]


def get_config():
    user = get_session('user')
    if user not in config:
        return redirect(url_for('login'))
    return config[user]


def get(key):
    return get_config()[key]


def update_config(key, value):
    config = get_config()
    config[key] = value


@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if db.session.query(User.id).filter_by(username=form.username.data).scalar() is None:
            return render_template('login.html', form=form, error="Invalid username or password")
        user = User.query.filter_by(username=form.username.data).first()
        if check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            session['user'] = user.username
            config[user.username] = {}
            if not os.path.exists(os.path.join('user_data/', user.username)):
                os.mkdir(os.path.join('user_data/', user.username))
            return redirect(url_for('upload'))
        return render_template('login.html', form=form, error='Invalid username or password')
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


def load_config():
    # read saved config
    CONFIG_FILE = config[get_session('user')]['config_file']
    conf = config_reader.read_config(CONFIG_FILE)

    # update files and df in config dict
    update_config('file', conf['PATHS']['file'])
    update_config('train_file', conf['PATHS']['train_file'])
    update_config('validation_file', conf['PATHS']['validation_file'])

    update_config('df', pd.read_csv(conf['PATHS']['file']))

    # retrieve values from config, assign_category does this
    df = get('df')
    df.reset_index(inplace=True, drop=True)
    categories, unique_values, default_list, frequent_values2frequency = assign_category(df, get_session('user'))

    data = df.head(SAMPLE_DATA_SIZE).T
    data.insert(0, 'Defaults', default_list.values())
    data.insert(0, '(most frequent, frequency)', frequent_values2frequency.values())
    data.insert(0, 'Unique Values', unique_values)
    data.insert(0, 'Category', categories)

    sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
    data.columns = list(
        itertools.chain(['Category', '#Unique Values', '(Most frequent, Frequency)', 'Defaults'], sample_column_names))

    update_config('data', data)
    update_config('defaults', dict(zip(get('data').index.tolist(), default_list.values())))
    update_config('category_list', categories)

    # target select
    target = conf['TARGET']['target']
    update_config('features', get('fs').create_tf_features(get('category_list'), target))
    update_config('target', target)

    config_writer.config = conf


def existing_data(form, user_configs):
    dataset_name = form['exisiting_files-train_file_exist']
    username = session['user']
    path = os.path.join(APP_ROOT, 'user_data', username, dataset_name)

    if 'exisiting_files-configuration' in form:
        config_name = form['exisiting_files-configuration']
        config[session['user']]['config_file'] = os.path.join(path, config_name, 'config.ini')
        load_config()
        return 'parameters'

    else:
        config_name = utils_config.define_new_config_file(dataset_name, APP_ROOT, username, config_writer)
        config[session['user']]['config_file'] = os.path.join(path, config_name, 'config.ini')
        if user_configs[dataset_name] and os.path.isfile(
                os.path.join(path, user_configs[dataset_name][0], 'config.ini')):
            reader = config_reader.read_config(os.path.join(path, user_configs[dataset_name][0], 'config.ini'))
            copyfile(os.path.join(path, user_configs[dataset_name][0], 'config.ini'),
                     os.path.join(path, config_name, 'config.ini'))
            filename = reader['PATHS']['file']
        elif os.path.isfile(os.path.join(path, dataset_name + '.csv')):
            filename = os.path.join(path, dataset_name + '.csv')
        else:
            filename = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and '.csv' in f][0]

        update_config('file', os.path.join(path, filename))
        config_writer.add_item('PATHS', 'file', os.path.join(path, filename))
        config_writer.write_config(config[get_session('user')]['config_file'])
        return 'slider'


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    config[session['user']] = {}
    global config_writer
    config_writer = ConfigWriter()
    form = UploadForm()
    user_dataset, user_configs, parameters_configs = utils_custom.get_configs_files(APP_ROOT, session['user'])
    form.exisiting_files.train_file_exist.choices = user_dataset
    if form.validate_on_submit():
        if form.is_existing.data:
            return redirect(url_for(existing_data(request.form, user_configs)))
        else:
            if form.new_files.train_file.data == '':
                form = UploadForm()
                return render_template('upload_file_form.html', form=form, page=0, user_configs=user_configs,
                                       parameters=parameters_configs)
            new_config(form, APP_ROOT, session['user'], config_writer)
        if not 'validation_file' in get_config() or not os.path.isfile(get('train_file')):
            return redirect(url_for('slider'))
        else:
            return redirect(url_for('feature'))
    flash_errors(form)
    if not user_configs:
        form = UploadNewForm()
        return render_template('upload_file_new_form.html', form=form, page=0)
    return render_template('upload_file_form.html', form=form, page=0, user_configs=user_configs,
                           parameters=parameters_configs)


@app.route('/upload_new', methods=['GET', 'POST'])
@login_required
def upload_new():
    form = UploadNewForm()
    if form.validate_on_submit():
        if form.new_files.train_file.data == '':
            return render_template('upload_file_new_form.html', form=form, page=0)
        new_config(form, APP_ROOT, session['user'], config_writer)
        if not 'validation_file' in get_config():
            return redirect(url_for('slider'))
        else:
            return redirect(url_for('feature'))
    flash_errors(form)
    return render_template('upload_file_new_form.html', form=form, page=0)


def new_config(form, APP_ROOT, username, config_writer):
    ext = form.new_files.train_file.data.filename.split('.')[-1]
    dataset_name = form.new_files.train_file.data.filename.split('.' + ext)[0]

    if os.path.isdir(os.path.join(APP_ROOT, 'user_data', session['user'], dataset_name)):
        dataset_name = utils_custom.generate_dataset_name(APP_ROOT, session['user'], dataset_name)

    config_name = utils_config.define_new_config_file(dataset_name, APP_ROOT, username, config_writer)
    config[username]['config_file'] = utils_config.create_config(username, APP_ROOT, dataset_name, config_name)
    path = os.path.join(APP_ROOT, 'user_data', username, dataset_name)

    save_filename(path, form.new_files.train_file, 'train_file', dataset_name)
    config_writer.add_item('PATHS', 'train_file', os.path.join(path, form.new_files.train_file.data.filename))

    config_writer.add_item('PATHS', 'file', os.path.join(path, form.new_files.train_file.data.filename))
    update_config('file', os.path.join(path, form.new_files.train_file.data.filename))

    if not isinstance(form.new_files.test_file.data, str):
        ext = form.new_files.test_file.data.filename.split('.')[-1]
        test_file = form.new_files.test_file.data.filename.split('.' + ext)[0]
        save_filename(path, form.new_files.test_file, 'validation_file', test_file)
        config_writer.add_item('PATHS', 'validation_file', os.path.join(path, form.new_files.test_file.data.filename))
        config_writer.write_config(config[get_session('user')]['config_file'])
        return redirect(url_for('feature'))

    config_writer.write_config(config[get_session('user')]['config_file'])


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    form = Submit(id="form")
    if form.validate_on_submit():
        update_config('split_df', request.form['percent'])
        return redirect(url_for('feature'))
    return render_template("slider.html", form=form, page=1)


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    # if 'df' not in get_config():x
    update_config('df', pd.read_csv(get('file')))
    df = get('df')
    df.reset_index(inplace=True, drop=True)
    categories, unique_values, default_list, frequent_values2frequency = assign_category(df, get_session(
        'user'))
    data = df.head(SAMPLE_DATA_SIZE).T
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
        cat_columns, default_values = utils_features.reorder_request(json.loads(request.form['default_featu']),
                                                                     json.loads(request.form['cat_column']),
                                                                     json.loads(request.form['default_column']),
                                                                     get('df').keys())
        update_config('category_list', cat_columns)
        get('data').Category = get('category_list')
        get('data').Defaults = default_values
        update_config('defaults', dict(zip(get('data').index.tolist(), default_values)))
        get('fs').update(get('category_list'), dict(zip(get('data').index.tolist(), default_values)))
        CONFIG_FILE = config[get_session('user')]['config_file']
        utils_custom.save_features_changes(CONFIG_FILE, get('data'), config_writer, categories)

        return redirect(url_for('target'))

    return render_template("feature_selection.html", name='Dataset features selection',
                           data=get('data'),
                           cat=categories, form=form, page=2)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    data = get('data')
    CONFIG_FILE = config[get_session('user')]['config_file']
    if form.validate_on_submit():
        target = json.loads(request.form['selected_row'])[0]
        update_config('features', get('fs').create_tf_features(get('category_list'), target))
        update_config('target', target)
        config_writer.add_item('TARGET', 'target', target)

        target_type = get('data').Category[get('target')]
        if target_type == 'range':
            new_categ_list = []
            for categ, feature in zip(get_config()['category_list'], get_config()['df'].columns):
                new_categ_list.append(categ if feature != target else 'categorical')
            update_config('category_list', new_categ_list)
            get('data').Category = get('category_list')
            get('fs').update(get('category_list'), dict(zip(get('data').index.tolist(), get('data').Defaults)))
        if 'split_df' in get_config():
            split_train_test(get('split_df'))
            config_writer.add_item('SPLIT_DF', 'split_df', get('split_df'))
        config_writer.write_config(CONFIG_FILE)
        return redirect(url_for('parameters'))
    target_selected = 'None'
    if 'TARGET' in config_reader.read_config(CONFIG_FILE).keys():
        target_selected = config_reader.read_config(CONFIG_FILE)['TARGET']['target']

    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data, page=3,
                           target_selected=target_selected)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    target_type = get('data').Category[get('target')]
    CONFIG_FILE = config[get_session('user')]['config_file']
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
    number_inputs = len(
        [get('data').Category[i] for i in range(len(get('data').Category)) if get('data').Category[i] != 'none']) - 1
    target_type = get('data').Category[get('target')]
    number_outputs = 1 if target_type == 'numerical' else len(
        get('fs').cat_unique_values_dict[get('target')])  # TODO fix
    num_samples = len(pd.read_csv(get('train_file')).index)
    utils_custom.get_defaults_param_form(form, CONFIG_FILE, number_inputs, number_outputs, num_samples, config_reader)
    return render_template('parameters.html', form=form, page=4)


@app.route('/run', methods=['GET', 'POST'])
def run():
    target_type = get('data').Category[get('target')]
    labels = utils_features.get_target_labels(get('target'), target_type, get('fs'))
    CONFIG_FILE = config[get_session('user')]['config_file']
    features = get('defaults')
    target = get('target')
    dict_types, categoricals = utils_run.get_dictionaries(get('defaults'), get('category_list'), get('fs'),
                                                          get('target'))
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    # checkpoints = utils_run.get_acc(directory, config_writer, CONFIG_FILE)
    checkpoints = utils_run.get_eval_results(directory, config_writer, CONFIG_FILE)
    sfeatures = features.copy()
    sfeatures.pop(target)
    if not session['user'] + '_' + CONFIG_FILE in ports.keys():
        port = utils_run.find_free_port()
        ports[session['user'] + '_' + CONFIG_FILE] = port
        tboard_thread = threading.Thread(name='tensor_board',
                                         target=lambda: threads.tensor_board_thread(CONFIG_FILE, port, config_reader))
        tboard_thread.setDaemon(True)
        tboard_thread.start()

    if request.method == 'POST':
        if request.form['action'] == 'run':
            dtypes = get('fs').group_by(get('category_list'))
            all_params_config = config_reader.read_config(CONFIG_FILE)
            r_thread = Process(
                target=lambda: threads.run_thread(all_params_config, get('features'), get('target'),
                                                  labels, get('defaults'), dtypes), name='run')
            r_thread.daemon = True
            r_thread.start()
            processes[get_session('user')] = r_thread
        else:
            threads.pause_threads(get_session('user'), processes)
        return jsonify(True)

    return render_template('run.html', page=5, features=sfeatures, target=get('target'),
                           types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                           checkpoints=checkpoints, port=ports[session['user'] + '_' + CONFIG_FILE])


@app.route('/delete', methods=['POST'])
def delete():
    features = get('defaults')
    target = get('target')
    CONFIG_FILE = config[get_session('user')]['config_file']
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    sfeatures = features.copy()
    sfeatures.pop(target)
    all_params_config = config_reader.read_config(CONFIG_FILE)
    del_id = request.get_json()['deleteID']
    paths = [del_id] if del_id != 'all' else [d for d in os.listdir(all_params_config.export_dir()) if
                                              os.path.isdir(os.path.join(all_params_config.export_dir(), d))]
    for p in paths:
        gfile.DeleteRecursively(os.path.join(all_params_config.export_dir(), p))
    checkpoints = utils_run.get_eval_results(directory, config_writer, CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/refresh', methods=['GET'])
@login_required
def refresh():
    CONFIG_FILE = config[get_session('user')]['config_file']
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    checkpoints = utils_run.get_eval_results(directory, config_writer, CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/predict', methods=['POST'])
def predict():
    target_type = get('data').Category[get('target')]
    features = get('defaults')
    target = get('target')
    CONFIG_FILE = config[get_session('user')]['config_file']

    new_features = {}
    for k, v in features.items():
        if k not in get('fs').group_by(get('category_list'))['none']:
            new_features[k] = request.form[k] if k != target else features[k]

    all_params_config = config_reader.read_config(CONFIG_FILE)
    all_params_config.set('PATHS', 'checkpoint_dir',
                          os.path.join(all_params_config.export_dir(), request.form['radiob']))
    labels = None if target_type == 'numerical' else get('fs').cat_unique_values_dict[get('target')]
    dtypes = get('fs').group_by(get('category_list'))
    r_thread = Process(target=lambda: predict_thread(all_params_config, get('features'), get('target'),
                                                     labels, get('defaults'), dtypes, new_features,
                                                     get('df')), name='predict')
    r_thread.daemon = True
    r_thread.start()
    r_thread.join()
    final_pred = return_dict['output']
    if final_pred is None:
        flash('Model\'s structure does not match the new parameter configuration', 'danger')
        final_pred = ''

    return jsonify(prediction=final_pred)


@app.route('/stream')
@login_required
def stream():
    config = get_config()['config_file'].split('/')
    logfile = os.path.join(APP_ROOT, 'user_data', session['user'], config[-3], config[-2], 'log', 'tensorflow.log')
    def generate():
        import tailer
        while not os.path.isfile(logfile):
            time.sleep(2)
        while True:
            for line in tailer.follow(open(logfile)):
                print(line)
                if line is not None:
                    yield line+'\n'

    return app.response_class(generate(), mimetype='text/plain')


@app.route('/')
def main():
    return redirect(url_for('login'))


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(percent):
    dataset_file = get('file')
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(percent)
    dataset = get('df')
    counts = dataset[get('target')].value_counts()
    dataset = dataset[dataset[get('target')].isin(counts[counts > 1].index)]
    target = dataset[[get('target')]]
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=target, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)

    update_config('train_file', train_file)
    update_config('validation_file', validation_file)

    config_writer.add_item('PATHS', 'train_file', train_file)
    config_writer.add_item('PATHS', 'validation_file', validation_file)


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)


def predict_thread(all_params_config, features, target, labels, defaults, dtypes, new_features, df):
    runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
    return_dict['output'] = runner.predict(new_features, target, df)


def assign_category(df, username):
    fs = FeatureSelection(df)
    update_config('fs', fs)
    feature_dict = fs.feature_dict()
    unique_values = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    CONFIG_FILE = config[username]['config_file']
    if 'COLUMN_CATEGORIES' in config_reader.read_config(CONFIG_FILE).keys():
        category_list = []
        for key in df.columns:
            category_list.append(config_reader.read_config(CONFIG_FILE)['COLUMN_CATEGORIES'][key])
    default_list = fs.defaults
    frequent_values2frequency = fs.frequent_values2frequency
    return category_list, unique_values, default_list, frequent_values2frequency


def save_filename(target, dataset_form_field, dataset_type, dataset_name):
    dataset_form_field.data.filename = dataset_name + '.csv'
    dataset_file = dataset_form_field.data
    if dataset_file:
        dataset_filename = secure_filename(dataset_file.filename)
        destination = os.path.join(target, dataset_filename)
        dataset_file.save(destination)
        update_config(dataset_type, destination)
    return True


db.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

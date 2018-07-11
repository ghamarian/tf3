import itertools
import json
import logging
import os
import pandas as pd
import psutil
import threading
import time
import utils_run
import utils_custom
from config import config_reader
from config.config_writer import ConfigWriter
from db import db
from feature_selection import FeatureSelection
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, login_user, login_required, logout_user
from forms.login_form import LoginForm
from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
from forms.run_form import RunForm
from forms.submit_form import Submit
from forms.upload_form import UploadForm, UploadNewForm
from multiprocessing import Process
from pprint import pprint
from runner import Runner
from sklearn.model_selection import train_test_split
from user import User
from utils import copyfile
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

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

# users = {}
# usertest = User('usertest2', generate_password_hash('test12345678', method='sha256'), 'test@test.com')
# users['usertest2'] = usertest

login_manager = LoginManager()
login_manager.init_app(app)


# login_manager.login_view = 'upload'


def get_session(user_id):
    with app.app_context():
        if user_id not in session:
            # session[user_id] = uuid.uuid4()
            return redirect(url_for('login'))
        return session[user_id]


def get_config():
    user = get_session('user')
    if user not in config:
        return redirect(url_for('login'))
    return config[user]


def create_config(dataset, config_name):
    user = get_session('user')
    path = APP_ROOT + '/user_data/' + user + '/' + dataset + '/' + config_name
    os.makedirs(path, exist_ok=True)
    copyfile('config/default_config.ini', path + '/config.ini')
    config[user]['config_file'] = path + '/config.ini'


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


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    user_dataset, user_configs, parameters_configs = utils_custom.get_configs_files(APP_ROOT, session['user'])
    form.exisiting_files.train_file_exist.choices = user_dataset
    if form.validate_on_submit():
        if form.is_existing.data:
            dataset_name = request.form['exisiting_files-train_file_exist']
            if 'exisiting_files-configuration' in request.form:
                config_name = request.form['exisiting_files-configuration']
            else:
                config_name = define_new_config_file(dataset_name, APP_ROOT, session['user'])
            config[session['user']]['config_file'] = os.path.join('user_data', session['user'], dataset_name,
                                                                  config_name, 'config.ini')
            train_file_name = os.path.join('user_data', session['user'], dataset_name, dataset_name + '-train.csv')
            update_config('train_file', os.path.join(APP_ROOT, train_file_name))
            config_writer.add_item('PATHS', 'train_file', os.path.join(APP_ROOT, train_file_name))
            test_file_name = os.path.join('user_data', session['user'], dataset_name, dataset_name + '-validation.csv')
            update_config('validation_file', os.path.join(APP_ROOT, test_file_name))
            config_writer.add_item('PATHS', 'validation_file', os.path.join(APP_ROOT, test_file_name))
            target = os.path.join(APP_ROOT, 'user_data', session['user'], dataset_name, config_name)
            update_config_checkpoints(config_writer, target)
        else:
            new_config(form, APP_ROOT, session['user'], config_writer)
        if not 'validation_file' in get_config():
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
        new_config(form, APP_ROOT, session['user'], config_writer)
        if not 'validation_file' in get_config():
            return redirect(url_for('slider'))
        else:
            return redirect(url_for('feature'))
    flash_errors(form)
    return render_template('upload_file_new_form.html', form=form, page=0)


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    form = Submit(id="form")
    if form.validate_on_submit():
        update_config('split_df', request.form['percent'])
        return redirect(url_for('feature'))
    return render_template("slider.html", form=form, page=1)


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    # if 'df' not in get_config():
    update_config('df', pd.read_csv(get('train_file')))

    df = get('df')
    df.reset_index(inplace=True, drop=True)
    categories, unique_values, default_list, frequent_values2frequency = assign_category(df)

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
        update_config('category_list', json.loads(request.form['cat_column']))
        default_values = json.loads(request.form['default_column'])
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
    number_outputs = 1 if target_type == 'numerical' else len(get('fs').cat_unique_values_dict[get('target')]) #TODO fix
    num_samples = len(get('df').index)

    utils_custom.get_defaults_param_form(form, CONFIG_FILE, number_inputs, number_outputs, num_samples, config_reader)
    return render_template('parameters.html', form=form, page=4)


@app.route('/run', methods=['GET', 'POST'])
def run():
    form = RunForm()
    target_type = get('data').Category[get('target')]
    labels = get_target_labels(get('target'), target_type, get('fs'))
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
    if not session['user'] in ports.keys():
        port = utils_run.find_free_port()
        ports[session['user']] = port
        tboard_thread = threading.Thread(name='tensor_board', target=lambda: tensor_board_thread(CONFIG_FILE, port))
        tboard_thread.setDaemon(True)
        tboard_thread.start()

    if form.validate_on_submit():
        dtypes = get('fs').group_by(get('category_list'))
        all_params_config = config_reader.read_config(CONFIG_FILE)
        r_thread = Process(target=lambda: run_thread(all_params_config, get('features'), get('target'),
                                                     labels, get('defaults'), dtypes), name='run')
        r_thread.daemon = True
        r_thread.start()
        processes[get_session('user')] = r_thread

        return render_template('run.html', form=form, running=1, page=5, features=sfeatures, target=get('target'),
                               types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                               checkpoints=checkpoints, port=ports[session['user']])

    running = 1 if get_session('user') in processes.keys() and not isinstance(processes[get_session('user')],
                                                                              str) else 0
    return render_template('run.html', form=form, running=running, page=5, features=sfeatures, target=get('target'),
                           types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                           checkpoints=checkpoints, port=ports[session['user']])


@app.route('/pause', methods=['GET', 'POST'])
def pause():
    import psutil
    p = processes[get_session('user')] if get_session('user') in processes.keys() else None
    if not isinstance(p, str) and p:
        pid = p.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print
            "child", child
            child.kill()
        parent.kill()
        processes[get_session('user')] = ''
    return redirect(url_for('run'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = RunForm()
    target_type = get('data').Category[get('target')]
    features = get('defaults')
    target = get('target')
    CONFIG_FILE = config[get_session('user')]['config_file']

    dict_types, categoricals = utils_run.get_dictionaries(get('defaults'), get('category_list'), get('fs'),
                                                          get('target'))
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']

    checkpoints = utils_run.get_eval_results(directory, config_writer, CONFIG_FILE)
    running = 1 if get_session('user') in processes.keys() and not isinstance(processes[get_session('user')],
                                                                              str) else 0
    if request.method == 'POST':
        new_features = {}
        for k, v in features.items():
            if k not in get('fs').group_by(get('category_list'))['none']:
                new_features[k] = request.form[k] if k != target else features[k]

        all_params_config = config_reader.read_config(CONFIG_FILE)
        all_params_config.set('PATHS', 'checkpoint_dir',
                              os.path.join(all_params_config.export_dir(), request.form['radiob']))
        labels = None if target_type == 'numerical' else get('fs').cat_unique_values_dict[get('target')]
        dtypes = get('fs').group_by(get('category_list'))
        runner = Runner(all_params_config, get('features'), get('target'), labels, get('defaults'), dtypes)
        final_pred = runner.predict(new_features, get('target'), get('df'))
        return render_template('run.html', form=form, running=running, page=5, features=new_features,
                               target=get('target'),
                               types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                               prediction=final_pred, checkpoints=checkpoints)
    sfeatures = features.copy()
    sfeatures.pop(target)
    return render_template('run.html', form=form, running=running, page=5, features=sfeatures, target=get('target'),
                           types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                           checkpoints=checkpoints)


@app.route('/')
def main():
    return redirect(url_for('login'))


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
    # train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=target)
    # train_df, test_df = train_test_split(dataset, test_size=0.8, stratify=target)
    train_df, test_df = train_test_split(dataset, test_size=test_size)
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
    unique_values = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    CONFIG_FILE = config[get_session('user')]['config_file']
    if 'COLUMN_CATEGORIES' in config_reader.read_config(CONFIG_FILE).keys():
        category_list = []
        for key in df.columns:
            category_list.append(config_reader.read_config(CONFIG_FILE)['COLUMN_CATEGORIES'][key])

    default_list = fs.defaults
    frequent_values2frequency = fs.frequent_values2frequency
    return category_list, unique_values, default_list, frequent_values2frequency


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


def tensor_board_thread(CONFIG_FILE, port):
    # TODO testing to multiuser
    config_path = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    logging.debug('Starting tensor board')
    time.sleep(3)
    os.system("tensorboard --logdir=" + config_path + "  --port=" + port)
    logging.debug('Exiting tensor board')


def run_thread(all_params_config, features, target, labels, defaults, dtypes):
    runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
    runner.run()


def create_config(dataset, config_name):
    user = get_session('user')
    path = APP_ROOT + '/user_data/' + user + '/' + dataset + '/' + config_name
    os.makedirs(path, exist_ok=True)
    copyfile('config/default_config.ini', path + '/config.ini')
    config[user]['config_file'] = path + '/config.ini'


def get_target_labels(target, target_type, fs):
    # TODO labels if target type is a RANGE, BOOL, ...
    if target_type == 'categorical' or target_type == 'hash':
        return fs.cat_unique_values_dict[target]
    elif target_type == 'range':
        return [str(a) for a in list(range(min(fs.df[target].values), max(fs.df[target].values)))]
    return None

def update_config_checkpoints(config_writer, target):
    config_writer.add_item('PATHS', 'checkpoint_dir', os.path.join(target, 'checkpoints/'))
    config_writer.add_item('PATHS', 'export_dir', os.path.join(target, 'checkpoints/export/best_exporter'))
    config_writer.add_item('PATHS', 'log_dir', os.path.join(target, 'log/'))

def define_new_config_file(dataset_name, APP_ROOT, username):
    config_name = utils_custom.generate_config_name(APP_ROOT, username, dataset_name)
    target = os.path.join(APP_ROOT, 'user_data', username, dataset_name, config_name)
    update_config_checkpoints(config_writer, target)
    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.join(target, 'log/'), exist_ok=True)
    create_config(dataset_name, config_name)
    return config_name


def new_config(form, APP_ROOT, username, config_writer):
    ext = form.new_files.train_file.data.filename.split('.')[-1]
    dataset_name = form.new_files.train_file.data.filename.split('.' + ext)[0]
    config_name = define_new_config_file(dataset_name, APP_ROOT, username)
    create_config(dataset_name, config_name)
    target_ds = os.path.join(APP_ROOT, 'user_data', username, dataset_name)
    save_file(target_ds, form.new_files.train_file, 'train_file')
    save_file(target_ds, form.new_files.test_file, 'validation_file')
    # TODO check if files exists
    if not 'validation_file' in get_config() and not isinstance(form.new_files.train_file.data,
                                                                str) and not isinstance(form.new_files.test_file.data,
                                                                                        str):
        target = os.path.join(APP_ROOT, 'user_data', username, dataset_name, config_name)
        config_writer.add_item('PATHS', 'train_file', os.path.join(target, form.new_files.train_file.data.filename))
        config_writer.add_item('PATHS', 'validation_file', os.path.join(target, form.new_files.test_file.data.filename))
        return redirect(url_for('feature'))



db.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

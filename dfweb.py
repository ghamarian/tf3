import itertools
import json
import logging
import os
import pandas as pd
import threading
import time
from feature_selection import FeatureSelection
from utilities import utils_run, utils_custom, threads, utils_features, upload_util, login_utils
from config import config_reader
from db import db
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, login_user, login_required, logout_user
from forms.login_form import LoginForm
from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
from forms.submit_form import Submit
from multiprocessing import Process
from pprint import pprint
from sklearn.model_selection import train_test_split
from user import User
from runner import Runner
from tensorflow.python.platform import gfile
from multiprocessing import Manager
from session import Session

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

sess = Session(app)
stop_event = threading.Event()
processes = {}
ports = {}

login_manager = LoginManager()
login_manager.init_app(app)

return_dict = Manager().dict()


@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if not login_utils.checklogin(form.username.data, form.password.data, form.remember.data, login_user, session,
                                      sess):
            return render_template('login.html', form=form, error='Invalid username or password')
        return redirect(url_for('upload'))
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    sess.reset_user()
    user_dataset, user_configs, param_configs = utils_custom.get_configs_files(APP_ROOT, session['user'])
    form, name_form = upload_util.create_form(user_configs, user_dataset)
    if form.validate_on_submit():
        url_redirect = upload_util.redirect(request, form, user_configs, session['user'], APP_ROOT, sess)
        if url_redirect: return redirect(url_for(url_redirect))
    flash_errors(form)
    return render_template(name_form, form=form, page=0, user_configs=user_configs, parameters=param_configs)


@app.route('/slider', methods=['GET', 'POST'])
def slider():
    form = Submit(id="form")
    if form.validate_on_submit():
        sess.set('split_df', request.form['percent'])
        return redirect(url_for('feature'))
    return render_template("slider.html", form=form, page=1)


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    sess.set('df', pd.read_csv(sess.get('file')))
    df = sess.get('df')
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

    sess.set('data', data)

    form = Submit()
    if form.validate_on_submit():
        cat_columns, default_values = utils_features.reorder_request(json.loads(request.form['default_featu']),
                                                                     json.loads(request.form['cat_column']),
                                                                     json.loads(request.form['default_column']),
                                                                     sess.get('df').keys())
        sess.set('category_list', cat_columns)
        sess.get('data').Category = sess.get('category_list')
        sess.get('data').Defaults = default_values
        sess.set('defaults', dict(zip(sess.get('data').index.tolist(), default_values)))
        sess.get('fs').update(sess.get('category_list'), dict(zip(sess.get('data').index.tolist(), default_values)))
        CONFIG_FILE = sess.get('config_file')
        utils_custom.save_features_changes(CONFIG_FILE, sess.get('data'), sess.get_writer(), categories)

        return redirect(url_for('target'))

    return render_template("feature_selection.html", name='Dataset features selection',
                           data=sess.get('data'),
                           cat=categories, form=form, page=2)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    data = sess.get('data')
    CONFIG_FILE = sess.get('config_file')
    if form.validate_on_submit():
        target = json.loads(request.form['selected_row'])[0]
        sess.set('features', sess.get('fs').create_tf_features(sess.get('category_list'), target))
        sess.set('target', target)
        sess.get_writer().add_item('TARGET', 'target', target)

        target_type = sess.get('data').Category[sess.get('target')]
        if target_type == 'range':
            new_categ_list = []
            for categ, feature in zip(sess.get('category_list'), sess.get('df').columns):
                new_categ_list.append(categ if feature != target else 'categorical')
            sess.set('category_list', new_categ_list)
            sess.get('data').Category = sess.get('category_list')
            sess.get('fs').update(sess.get('category_list'),
                                  dict(zip(sess.get('data').index.tolist(), sess.get('data').Defaults)))
        if 'split_df' in sess.get_config():
            split_train_test(sess.get('split_df'))
            sess.get_writer().add_item('SPLIT_DF', 'split_df', sess.get('split_df'))
        sess.get_writer().write_config(CONFIG_FILE)
        return redirect(url_for('parameters'))
    target_selected = 'None'
    if 'TARGET' in config_reader.read_config(CONFIG_FILE).keys():
        target_selected = config_reader.read_config(CONFIG_FILE)['TARGET']['target']

    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data, page=3,
                           target_selected=target_selected)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    target_type = sess.get('data').Category[sess.get('target')]
    CONFIG_FILE = sess.get('config_file')
    if target_type == 'numerical':
        form = GeneralRegressorForm()
    else:
        form = GeneralClassifierForm()
    if form.validate_on_submit():
        pprint(request.form)
        sess.get_writer().populate_config(request.form)
        sess.get_writer().write_config(CONFIG_FILE)
        return redirect(url_for('run'))
    flash_errors(form)
    number_inputs = len(
        [sess.get('data').Category[i] for i in range(len(sess.get('data').Category)) if
         sess.get('data').Category[i] != 'none']) - 1
    target_type = sess.get('data').Category[sess.get('target')]
    number_outputs = 1 if target_type == 'numerical' else sess.get('data')['#Unique Values'][
        sess.get('target')]  # TODO fix
    num_samples = len(pd.read_csv(sess.get('train_file')).index)
    utils_custom.get_defaults_param_form(form, CONFIG_FILE, number_inputs, number_outputs, num_samples, config_reader)
    return render_template('parameters.html', form=form, page=4)


@app.route('/run', methods=['GET', 'POST'])
def run():
    target_type = sess.get('data').Category[sess.get('target')]
    labels = utils_features.get_target_labels(sess.get('target'), target_type, sess.get('fs'))
    CONFIG_FILE = sess.get('config_file')
    features = sess.get('defaults')
    target = sess.get('target')
    dict_types, categoricals = utils_run.get_dictionaries(sess.get('defaults'), sess.get('category_list'),
                                                          sess.get('fs'),
                                                          sess.get('target'))
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    # checkpoints = utils_run.get_acc(directory, config_writer, CONFIG_FILE)
    checkpoints = utils_run.get_eval_results(directory, sess.get_writer(), CONFIG_FILE)
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
            dtypes = sess.get('fs').group_by(sess.get('category_list'))
            all_params_config = config_reader.read_config(CONFIG_FILE)
            r_thread = Process(
                target=lambda: threads.run_thread(all_params_config, sess.get('features'), sess.get('target'),
                                                  labels, sess.get('defaults'), dtypes), name='run')
            r_thread.daemon = True
            r_thread.start()
            processes[sess.get_session('user')] = r_thread
        else:
            threads.pause_threads(sess.get_session('user'), processes)
        return jsonify(True)

    return render_template('run.html', page=5, features=sfeatures, target=sess.get('target'),
                           types=utils_custom.get_html_types(dict_types), categoricals=categoricals,
                           checkpoints=checkpoints, port=ports[session['user'] + '_' + CONFIG_FILE])


@app.route('/delete', methods=['POST'])
def delete():
    features = sess.get('defaults')
    target = sess.get('target')
    CONFIG_FILE = sess.get('config_file')
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    sfeatures = features.copy()
    sfeatures.pop(target)
    all_params_config = config_reader.read_config(CONFIG_FILE)
    del_id = request.get_json()['deleteID']
    paths = [del_id] if del_id != 'all' else [d for d in os.listdir(all_params_config.export_dir()) if
                                              os.path.isdir(os.path.join(all_params_config.export_dir(), d))]
    for p in paths:
        gfile.DeleteRecursively(os.path.join(all_params_config.export_dir(), p))
    checkpoints = utils_run.get_eval_results(directory, sess.get_writer(), CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/refresh', methods=['GET'])
@login_required
def refresh():
    CONFIG_FILE = sess.get('config_file')
    directory = config_reader.read_config(CONFIG_FILE).all()['export_dir']
    checkpoints = utils_run.get_eval_results(directory, sess.get_writer(), CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/predict', methods=['POST'])
def predict():
    target_type = sess.get('data').Category[sess.get('target')]
    features = sess.get('defaults')
    target = sess.get('target')
    CONFIG_FILE = sess.get('config_file')

    new_features = {}
    for k, v in features.items():
        if k not in sess.get('fs').group_by(sess.get('category_list'))['none']:
            new_features[k] = request.form[k] if k != target else features[k]

    all_params_config = config_reader.read_config(CONFIG_FILE)
    all_params_config.set('PATHS', 'checkpoint_dir',
                          os.path.join(all_params_config.export_dir(), request.form['radiob']))
    labels = None if target_type == 'numerical' else sess.get('fs').cat_unique_values_dict[sess.get('target')]
    dtypes = sess.get('fs').group_by(sess.get('category_list'))
    r_thread = Process(target=lambda: predict_thread(all_params_config, sess.get('features'), sess.get('target'),
                                                     labels, sess.get('defaults'), dtypes, new_features,
                                                     sess.get('df')), name='predict')
    r_thread.daemon = True
    r_thread.start()
    r_thread.join()
    final_pred = return_dict['output']
    if final_pred is None:
        flash('Model\'s structure does not match the new parameter configuration', 'danger')
        final_pred = ''

    return jsonify(prediction=str(final_pred))


@app.route('/stream')
@login_required
def stream():
    config = sess.get('config_file').split('/')
    logfile = os.path.join(APP_ROOT, 'user_data', session['user'], config[-3], config[-2], 'log', 'tensorflow.log')

    def generate():
        import tailer
        while not os.path.isfile(logfile):
            time.sleep(2)
        while True:
            for line in tailer.follow(open(logfile)):
                print(line)
                if line is not None:
                    yield line + '\n'

    return app.response_class(generate(), mimetype='text/plain')


@app.route('/')
def main():
    return redirect(url_for('login'))


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(percent):
    dataset_file = sess.get('file')
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(percent)
    dataset = sess.get('df')
    counts = dataset[sess.get('target')].value_counts()
    dataset = dataset[dataset[sess.get('target')].isin(counts[counts > 1].index)]
    target = dataset[[sess.get('target')]]
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=target, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)

    sess.set('train_file', train_file)
    sess.set('validation_file', validation_file)

    sess.get_writer().add_item('PATHS', 'train_file', train_file)
    sess.get_writer().add_item('PATHS', 'validation_file', validation_file)


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)


def predict_thread(all_params_config, features, target, labels, defaults, dtypes, new_features, df):
    runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
    return_dict['output'] = runner.predict(new_features, target, df)


def assign_category(df):
    fs = FeatureSelection(df)
    sess.set('fs', fs)
    feature_dict = fs.feature_dict()
    unique_values = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
    category_list = [feature_dict[key] for key in df.columns]
    CONFIG_FILE = sess.get('config_file')
    if 'COLUMN_CATEGORIES' in config_reader.read_config(CONFIG_FILE).keys():
        category_list = []
        for key in df.columns:
            category_list.append(config_reader.read_config(CONFIG_FILE)['COLUMN_CATEGORIES'][key])
    default_list = fs.defaults
    frequent_values2frequency = fs.frequent_values2frequency
    return category_list, unique_values, default_list, frequent_values2frequency


db.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

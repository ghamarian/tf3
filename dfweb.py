import json
import os
import time
from io import BytesIO
from utils import run_utils, upload_util, db_ops, feature_util, param_utils, preprocessing, config_ops, sys_ops
from config import config_reader
from database.db import db
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, login_user, login_required, logout_user
from forms.login_form import LoginForm
from forms.submit_form import Submit
from pprint import pprint
from user import User

from thread_handler import ThreadHandler
from session import Session

SAMPLE_DATA_SIZE = 5
WTF_CSRF_SECRET_KEY = os.urandom(42)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///username.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

sess = Session(app)
th = ThreadHandler()

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if not db_ops.checklogin(form.username.data, form.password.data, form.remember.data, login_user, session,
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
    user_dataset, user_configs, param_configs = config_ops.get_configs_files(APP_ROOT, session['user'])
    form, name_form = upload_util.create_form(user_configs, user_dataset)
    if form.validate_on_submit():
        if hasattr(form, 'exisiting_files') and form.is_existing.data:
            return redirect(
                url_for(upload_util.existing_data(request.form, user_configs, session['user'], sess, APP_ROOT)))
        elif not form.new_files.train_file.data == '':
            return redirect(
                url_for(config_ops.new_config(form.new_files.train_file.data, form.new_files.test_file.data, APP_ROOT,
                                              session['user'], sess)))
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
    sess.load_features()
    form = Submit()
    if form.validate_on_submit():
        old_cats = sess.get('category_list')
        cat_columns, default_values = feature_util.reorder_request(json.loads(request.form['default_featu']),
                                                                   json.loads(request.form['cat_column']),
                                                                   json.loads(request.form['default_column']),
                                                                   sess.get('df').keys())
        sess.update_new_features(cat_columns, default_values)
        feature_util.write_features(old_cats, sess.get('data'), sess.get_writer(),
                                    sess.get('config_file'))
        return redirect(url_for('target'))
    return render_template("feature_selection.html", name='Dataset features selection',
                           data=sess.get('data'), cat=sess.get('category_list'), form=form, page=2)


@app.route('/target', methods=['POST', 'GET'])
def target():
    form = Submit()
    if form.validate_on_submit():
        sess.set_target(json.loads(request.form['selected_row'])[0])
        if 'split_df' in sess.get_config():
            train_file, validation_file = preprocessing.split_train_test(sess.get('split_df'), sess.get('file'),
                                                                         sess.get('target'), sess.get('df'))
            sess.update_split(train_file, validation_file)
        sess.get_writer().write_config(sess.get('config_file'))
        return redirect(url_for('parameters'))
    reader = config_reader.read_config(sess.get('config_file'))
    target_selected = reader['TARGET']['target'] if 'TARGET' in reader.keys() else 'None'
    # filter hash and none columns
    data = sess.get('data')[(sess.get('data').Category !='hash') & (sess.get('data').Category !='none')]

    return render_template('target_selection.html', name="Dataset target selection", form=form, data=data,
                           page=3, target_selected=target_selected)


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    CONFIG_FILE = sess.get('config_file')
    form = param_utils.define_param_form(sess.get('data').Category[sess.get('target')])
    if form.validate_on_submit():
        pprint(request.form)
        sess.get_writer().populate_config(request.form)
        sess.get_writer().write_config(CONFIG_FILE)
        return redirect(url_for('run'))
    flash_errors(form)
    param_utils.get_defaults_param_form(form, CONFIG_FILE, sess.get('data'), sess.get('target'),
                                        sess.get('train_file'))
    return render_template('parameters.html', form=form, page=4)


@app.route('/run', methods=['GET', 'POST'])
def run():
    all_params_config = config_reader.read_config(sess.get('config_file'))
    logfile = os.path.join(all_params_config['PATHS']['log_dir'], 'tensorflow.log')
    try:
        sess.set('log_fp', open(logfile))
    except:
        open(logfile, 'a').close()
        sess.set('log_fp', open(logfile))

    labels = feature_util.get_target_labels(sess.get('target'), sess.get('data').Category[sess.get('target')],
                                            sess.get('fs'))

    export_dir = all_params_config.export_dir()
    checkpoints = run_utils.get_eval_results(export_dir, sess.get_writer(), sess.get('config_file'))
    th.run_tensor_board(session['user'], sess.get('config_file'))
    if request.method == 'POST':

        dtypes = sess.get('fs').group_by(sess.get('category_list'))
        th.handle_request(request.form['action'], all_params_config, sess.get('features'), sess.get('target'), labels,
                          sess.get('defaults'), dtypes, session['user'], request.form['resume_from'])
        return jsonify(True)
    dict_types, categoricals = run_utils.get_dictionaries(sess.get('defaults'), sess.get('category_list'),
                                                          sess.get('fs'), sess.get('target'))
    sfeatures = feature_util.remove_target(sess.get('defaults'), sess.get('target'))
    return render_template('run.html', page=5, features=sfeatures, target=sess.get('target'),
                           types=run_utils.get_html_types(dict_types), categoricals=categoricals,
                           checkpoints=checkpoints, port=th.get_port(session['user'], sess.get('config_file')))


@app.route('/predict', methods=['POST'])
def predict():
    new_features = feature_util.get_new_features(request.form, sess.get('defaults'), sess.get('target'),
                                                 sess.get('fs').group_by(sess.get('category_list'))['none'])
    all_params_config = config_reader.read_config(sess.get('config_file'))
    all_params_config.set('PATHS', 'checkpoint_dir',
                          os.path.join(all_params_config.export_dir(), request.form['radiob']))
    labels = feature_util.get_target_labels(sess.get('target'), sess.get('data').Category[sess.get('target')],
                                            sess.get('fs'))
    dtypes = sess.get('fs').group_by(sess.get('category_list'))
    final_pred = th.predict_estimator(all_params_config, sess.get('features'), sess.get('target'), labels,
                                      sess.get('defaults'), dtypes,
                                      new_features, sess.get('df'))

    return jsonify(prediction=str(final_pred))


@app.route('/explain', methods=['GET', 'POST'])
def explain():
    if request.method == 'POST':
        new_features = feature_util.get_new_features(request.form, sess.get('defaults'), sess.get('target'),
                                                     sess.get('fs').group_by(sess.get('category_list'))['none'])
        all_params_config = config_reader.read_config(sess.get('config_file'))
        all_params_config.set('PATHS', 'checkpoint_dir',
                              os.path.join(all_params_config.export_dir(), request.form['radiob']))
        labels = feature_util.get_target_labels(sess.get('target'), sess.get('data').Category[sess.get('target')],
                                                sess.get('fs'))
        dtypes = sess.get('fs').group_by(sess.get('category_list'))
        result = th.explain_estimator(all_params_config, sess.get('features'), sess.get('target'), labels,
                                      sess.get('defaults'), dtypes, new_features, sess.get('df'),
                                      sess.get('data').Category,
                                      int(request.form['num_feat']), int(request.form['top_labels']))
        if result is not None:
            fp = BytesIO(str.encode(result.as_html(show_table=True)))
            sess.set('explain_fp', fp)
            return jsonify(explanation='ok')

        return jsonify(explanation=str(result))
    else:
        return send_file(sess.get('explain_fp'),  mimetype='text/html')


@app.route('/delete', methods=['POST'])
@login_required
def delete():
    CONFIG_FILE = sess.get('config_file')
    all_params_config = config_reader.read_config(CONFIG_FILE)
    export_dir = all_params_config.export_dir()
    del_id = request.get_json()['deleteID']
    paths = [del_id] if del_id != 'all' else [d for d in os.listdir(export_dir) if
                                              os.path.isdir(os.path.join(export_dir, d))]
    sys_ops.delete_recursive(paths, export_dir)
    checkpoints = run_utils.get_eval_results(export_dir, sess.get_writer(), CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/delete_config', methods=['POST'])
@login_required
def delete_config():
    sys_ops.delete_configs(request.get_json()['config'], request.get_json()['dataset'], session['user'])
    user_dataset, user_configs, param_configs = config_ops.get_configs_files(APP_ROOT, session['user'])
    return jsonify(configs=user_configs, params=param_configs)


@app.route('/refresh', methods=['GET'])
@login_required
def refresh():
    CONFIG_FILE = sess.get('config_file')
    export_dir = config_reader.read_config(CONFIG_FILE).export_dir()
    checkpoints = run_utils.get_eval_results(export_dir, sess.get_writer(), CONFIG_FILE)
    return jsonify(checkpoints=checkpoints)


@app.route('/stream')
@login_required
def stream():
    return jsonify(data=sess.get('log_fp').read())


@app.route('/')
def main():
    return redirect(url_for('login'))


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)


db.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

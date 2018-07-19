import os
from config import config_reader
from utilities import utils_config, utils_custom
from shutil import copyfile
from werkzeug.utils import secure_filename
from forms.upload_form import UploadForm, UploadNewForm
from utilities import upload_util


def redirect(request, form, user_configs, username, APP_ROOT, sess):
    if hasattr(form, 'exisiting_files') and form.is_existing.data:
        return upload_util.existing_data(request.form, user_configs, username, sess, APP_ROOT)
    elif not form.new_files.train_file.data == '':
        if upload_util.new_config(form.new_files.train_file.data, form.new_files.test_file.data, APP_ROOT,
                                  username, sess):
            return 'slider'
        return 'feature'
    return None


def create_form(user_configs, user_dataset):
    if not user_configs:
        form = UploadNewForm()
        return form, 'upload_file_new_form.html'
    form = UploadForm()
    form.exisiting_files.train_file_exist.choices = user_dataset
    return form, 'upload_file_form.html'


def save_filename(target, dataset_form_field, dataset_type, dataset_name, sess):
    dataset_form_field.filename = dataset_name + '.csv'
    dataset_file = dataset_form_field
    if dataset_file:
        dataset_filename = secure_filename(dataset_file.filename)
        destination = os.path.join(target, dataset_filename)
        dataset_file.save(destination)
        sess.set(dataset_type, destination)
    return True


def existing_data(form, user_configs, username, sess, APP_ROOT):
    dataset_name = form['exisiting_files-train_file_exist']
    path = os.path.join(APP_ROOT, 'user_data', username, dataset_name)
    if 'exisiting_files-configuration' in form:
        config_name = form['exisiting_files-configuration']
        sess.set('config_file', os.path.join(path, config_name, 'config.ini'))
        sess.load_config()
        return 'parameters'
    else:
        config_name = utils_config.define_new_config_file(dataset_name, APP_ROOT, username, sess.get_writer())
        sess.set('config_file', os.path.join(path, config_name, 'config.ini'))
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
        sess.set('file', os.path.join(path, filename))
        sess.get_writer().add_item('PATHS', 'file', os.path.join(path, filename))
        sess.get_writer().write_config(sess.get('config_file'))
        return 'slider'


def new_config(train_form_file, test_form_file, APP_ROOT, username, sess):
    ext = train_form_file.filename.split('.')[-1]
    dataset_name = train_form_file.filename.split('.' + ext)[0]
    if os.path.isdir(os.path.join(APP_ROOT, 'user_data', username, dataset_name)):
        dataset_name = utils_custom.generate_dataset_name(APP_ROOT, username, dataset_name)

    config_name = utils_config.define_new_config_file(dataset_name, APP_ROOT, username, sess.get_writer())
    sess.set('config_file', utils_config.create_config(username, APP_ROOT, dataset_name, config_name))
    path = os.path.join(APP_ROOT, 'user_data', username, dataset_name)

    save_filename(path, train_form_file, 'train_file', dataset_name, sess)
    sess.get_writer().add_item('PATHS', 'train_file', os.path.join(path, train_form_file.filename))

    sess.get_writer().add_item('PATHS', 'file', os.path.join(path, train_form_file.filename))
    sess.set('file', os.path.join(path, train_form_file.filename))

    if not isinstance(test_form_file, str):
        ext = test_form_file.filename.split('.')[-1]
        test_file = test_form_file.filename.split('.' + ext)[0]
        save_filename(path, test_form_file, 'validation_file', test_file, sess)
        sess.get_writer().add_item('PATHS', 'validation_file', os.path.join(path, test_form_file.filename))
        sess.get_writer().write_config(sess.get('config_file'))
        return False
    sess.get_writer().write_config(sess.get('config_file'))
    return True

import os
from utils import sys_ops
import configparser
from tensorflow.python.platform import gfile
from utils import upload_util, sys_ops


def generate_config_name(app_root, username, dataset_name):
    user_configs = []
    if os.path.isdir(os.path.join(app_root, 'user_data', username, dataset_name)):
        user_configs = [a for a in os.listdir(os.path.join(app_root, 'user_data', username, dataset_name))
                        if os.path.isdir(os.path.join(app_root, 'user_data', username, dataset_name, a))]
    new_name = 'config_'
    cont = 1
    while new_name + str(cont) in user_configs:
        cont += 1
    return new_name + str(cont)


def create_config(username, APP_ROOT, dataset, config_name):
    path = APP_ROOT + '/user_data/' + username + '/' + dataset + '/' + config_name
    os.makedirs(path, exist_ok=True)
    sys_ops.copyfile('config/default_config.ini', path + '/config.ini')
    return path + '/config.ini'


def update_config_dir(config_writer, target):
    config_writer.add_item('PATHS', 'checkpoint_dir', os.path.join(target, 'checkpoints/'))
    config_writer.add_item('PATHS', 'export_dir', os.path.join(target, 'checkpoints/export/best_exporter'))
    config_writer.add_item('PATHS', 'log_dir', os.path.join(target, 'log/'))


def define_new_config_file(dataset_name, APP_ROOT, username, config_writer):
    config_name = generate_config_name(APP_ROOT, username, dataset_name)
    target = os.path.join(APP_ROOT, 'user_data', username, dataset_name, config_name)
    update_config_dir(config_writer, target)
    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.join(target, 'log/'), exist_ok=True)
    create_config(username, APP_ROOT, dataset_name, config_name)
    return config_name


def get_configs_files(app_root, username):
    user_configs = {}
    parameters_configs = {}
    dataset_form_exis = []
    path = os.path.join(app_root, 'user_data', username)
    user_datasets = [a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))]
    for user_dataset in user_datasets:
        user_configs[user_dataset] = [a for a in os.listdir(os.path.join(path, user_dataset)) if
                                      os.path.isdir(os.path.join(path, user_dataset, a))]
        # TODO parameters to show how information of configuration model
        for config_file in user_configs[user_dataset]:
            parameters_configs[user_dataset + '_' + config_file] = {}
            connfig = configparser.ConfigParser()
            connfig.read(os.path.join(path, user_dataset, config_file, 'config.ini'))
            if 'NETWORK' in connfig.sections():
                if 'BEST_MODEL' in connfig.sections():
                    parameters_configs[user_dataset + '_' + config_file]['acc'] = connfig.get('BEST_MODEL', 'max_acc')
                    parameters_configs[user_dataset + '_' + config_file]['loss'] = connfig.get('BEST_MODEL', 'min_loss')
                parameters_configs[user_dataset + '_' + config_file]['model'] = connfig.get('NETWORK', 'model_name')
            else:
                gfile.DeleteRecursively(os.path.join(path, user_dataset, config_file))

        user_configs[user_dataset] = [a for a in os.listdir(os.path.join(path, user_dataset)) if
                                      os.path.isdir(os.path.join(path, user_dataset, a))]
        dataset_form_exis.append((user_dataset, user_dataset))
    return dataset_form_exis, user_configs, parameters_configs


def new_config(train_form_file, test_form_file, APP_ROOT, username, sess):
    ext = train_form_file.filename.split('.')[-1]
    dataset_name = train_form_file.filename.split('.' + ext)[0]
    if os.path.isdir(os.path.join(APP_ROOT, 'user_data', username, dataset_name)):
        dataset_name = upload_util.generate_dataset_name(APP_ROOT, username, dataset_name)

    config_name = define_new_config_file(dataset_name, APP_ROOT, username, sess.get_writer())
    sess.set('config_file', create_config(username, APP_ROOT, dataset_name, config_name))
    path = os.path.join(APP_ROOT, 'user_data', username, dataset_name)

    sys_ops.save_filename(path, train_form_file, 'train_file', dataset_name, sess)
    sess.get_writer().add_item('PATHS', 'train_file', os.path.join(path, train_form_file.filename))
    sess.get_writer().add_item('PATHS', 'file', os.path.join(path, train_form_file.filename))
    sess.set('file', os.path.join(path, train_form_file.filename))
    if not isinstance(test_form_file, str):
        ext = test_form_file.filename.split('.')[-1]
        test_file = test_form_file.filename.split('.' + ext)[0]
        sys_ops.save_filename(path, test_form_file, 'validation_file', test_file, sess)
        sess.get_writer().add_item('PATHS', 'validation_file', os.path.join(path, test_form_file.filename))
        sess.get_writer().write_config(sess.get('config_file'))
        return 'feature'
    sess.get_writer().write_config(sess.get('config_file'))
    return 'slider'



import os
from utilities import utils_io


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
    utils_io.copyfile('config/default_config.ini', path + '/config.ini')
    return path + '/config.ini'


def update_config_checkpoints(config_writer, target):
    config_writer.add_item('PATHS', 'checkpoint_dir', os.path.join(target, 'checkpoints/'))
    config_writer.add_item('PATHS', 'export_dir', os.path.join(target, 'checkpoints/export/best_exporter'))
    config_writer.add_item('PATHS', 'log_dir', os.path.join(target, 'log/'))


def define_new_config_file(dataset_name, APP_ROOT, username, config_writer):
    config_name = generate_config_name(APP_ROOT, username, dataset_name)
    target = os.path.join(APP_ROOT, 'user_data', username, dataset_name, config_name)
    update_config_checkpoints(config_writer, target)
    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.join(target, 'log/'), exist_ok=True)
    create_config(username, APP_ROOT, dataset_name, config_name)
    return config_name
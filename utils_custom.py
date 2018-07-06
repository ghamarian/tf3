import os
import numpy as np
from config.config_writer import ConfigWriter


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


def get_html_types(dict_types):
    dict_html_types = {}
    for k, v in dict_types.items():
        dict_html_types[k] = "text" if v == 'categorical' else "number"
    return dict_html_types


def get_hidden_layers(INPUT_DIM, layers=2):
    hidden = [int(width) for width in np.exp(np.log(INPUT_DIM) * np.arange(layers - 1, 0, -1) / layers)]
    return ','.join(str(x) for x in hidden)


def get_configs_files(app_root, username):
    import configparser
    connfig = configparser.ConfigParser()
    user_configs = {}
    parameters_configs = {}
    dataset_form_exis = []
    path = os.path.join(app_root, 'user_data', username)
    user_datasets = [a for a in os.listdir(path) if
                     os.path.isdir(os.path.join(path, a))]
    for user_dataset in user_datasets:
        user_configs[user_dataset] = [a for a in os.listdir(os.path.join(path, user_dataset)) if
                                      os.path.isdir(os.path.join(path, user_dataset, a))]
        # TODO parameters to show how information of configuration model
        for config_file in user_configs[user_dataset]:
            connfig.read(os.path.join(path, user_dataset, config_file, 'config.ini'))
            if 'NETWORK' in connfig.keys():
                parameters_configs[user_dataset + '_' + config_file] = connfig.get('NETWORK', 'model_name')
        dataset_form_exis.append((user_dataset, user_dataset))
    return dataset_form_exis, user_configs, parameters_configs


def save_features_changes(CONFIG_FILE, data, config_writer):
    for label in data.index:
        cat = data.Category[label] if data.Category[label] != 'range' else 'int-range'
        config_writer.add_item('COLUMN_CATEGORIES', label, cat)
    config_writer.write_config(CONFIG_FILE)

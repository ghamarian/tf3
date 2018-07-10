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


def get_hidden_layers(INPUT_DIM, OUTUPUT_DIM, num_samples, alpha=2):
    size = num_samples/(alpha * (INPUT_DIM + OUTUPUT_DIM))
    return str(int(round(size)))


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
            parameters_configs[user_dataset + '_' + config_file] = {}
            connfig.read(os.path.join(path, user_dataset, config_file, 'config.ini'))
            if 'BEST_MODEL' in connfig.keys():
                parameters_configs[user_dataset + '_' + config_file]['acc'] = connfig.get('BEST_MODEL', 'max_acc')
                parameters_configs[user_dataset + '_' + config_file]['loss'] = connfig.get('BEST_MODEL', 'min_loss')
            if 'NETWORK' in connfig.keys():
                parameters_configs[user_dataset + '_' + config_file]['model'] = connfig.get('NETWORK', 'model_name')
        dataset_form_exis.append((user_dataset, user_dataset))
    return dataset_form_exis, user_configs, parameters_configs


def save_features_changes(CONFIG_FILE, data, config_writer, categories):
    for label, categories in zip(data.index, categories):
        cat = data.Category[label] if data.Category[label] != 'range' else 'int-range'
        if 'none' in cat:
            cat = 'none' + '-' + categories if 'none' not in categories else categories
        config_writer.add_item('COLUMN_CATEGORIES', label, cat)
    config_writer.write_config(CONFIG_FILE)


def get_defaults_param_form(form, CONFIG_FILE, number_inputs, number_outputs, num_samples, config_reader):
    form.network.form.hidden_layers.default = get_hidden_layers(number_inputs, number_outputs, num_samples)
    form.network.form.process()
    if 'EXPERIMENT' in config_reader.read_config(CONFIG_FILE).keys():
        form.experiment.form.keep_checkpoint_max.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT'][
            'keep_checkpoint_max']
        form.experiment.form.save_checkpoints_steps.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT'][
            'save_checkpoints_steps']
        form.experiment.form.initialize_with_checkpoint.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT'][
            'initialize_with_checkpoint']
        form.experiment.form.save_summary_steps.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT'][
            'save_summary_steps']
        form.experiment.form.throttle.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT']['throttle']
        form.experiment.form.validation_batch_size.default = config_reader.read_config(CONFIG_FILE)['EXPERIMENT'][
            'validation_batch_size']
    if 'NETWORK' in config_reader.read_config(CONFIG_FILE).keys():
        form.network.form.hidden_layers.default = config_reader.read_config(CONFIG_FILE)['NETWORK']['hidden_layers']
        form.network.form.model_name.default = config_reader.read_config(CONFIG_FILE)['NETWORK']['model_name']
    if 'CUSTOM_MODEL' in config_reader.read_config(CONFIG_FILE).keys():
        form.custom_model.form.custom_model_path.default = config_reader.read_config(CONFIG_FILE)['CUSTOM_MODEL'][
            'custom_model_path']
    if 'TRAINING' in config_reader.read_config(CONFIG_FILE).keys():
        form.training.form.num_epochs.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['num_epochs']
        form.training.form.batch_size.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['batch_size']
        form.training.form.optimizer.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['optimizer']
        form.training.form.learning_rate.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['learning_rate']
        form.training.form.l1_regularization.default = config_reader.read_config(CONFIG_FILE)['TRAINING'][
            'l1_regularization']
        form.training.form.l2_regularization.default = config_reader.read_config(CONFIG_FILE)['TRAINING'][
            'l2_regularization']
        form.training.form.dropout.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['dropout']
        form.training.form.activation_fn.default = config_reader.read_config(CONFIG_FILE)['TRAINING']['activation_fn']
    form.network.form.process()
    form.experiment.form.process()
    form.training.form.process()
    form.custom_model.form.process()

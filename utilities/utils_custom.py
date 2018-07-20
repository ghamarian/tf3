import os
from tensorflow.python.platform import gfile
import configparser

def generate_dataset_name(app_root, username, dataset_name):
    user_datasets = []
    if os.path.isdir(os.path.join(app_root, 'user_data', username)):
        user_datasets = [a for a in os.listdir(os.path.join(app_root, 'user_data', username))
                         if os.path.isdir(os.path.join(app_root, 'user_data', username, a))]
    cont = 1
    while dataset_name + '_' + str(cont) in user_datasets:
        cont += 1
    new_dataset_name = dataset_name + '_' + str(cont)
    return new_dataset_name


def get_html_types(dict_types):
    dict_html_types = {}
    for k, v in dict_types.items():
        dict_html_types[k] = "text" if v == 'categorical' else "number"
    return dict_html_types


def get_hidden_layers(INPUT_DIM, OUTUPUT_DIM, num_samples, alpha=2):
    size = num_samples / (alpha * (INPUT_DIM + OUTUPUT_DIM))
    return str(int(round(size)))


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


def get_defaults_param_form(form, CONFIG_FILE, number_inputs, number_outputs, num_samples, config_reader):
    form.network.form.hidden_layers.default = get_hidden_layers(number_inputs, number_outputs, num_samples)
    form.network.form.process()
    reader = config_reader.read_config(CONFIG_FILE)
    if 'EXPERIMENT' in reader.keys():
        form.experiment.form.keep_checkpoint_max.default = reader['EXPERIMENT']['keep_checkpoint_max']
        form.experiment.form.save_checkpoints_steps.default = reader['EXPERIMENT']['save_checkpoints_steps']
        form.experiment.form.save_summary_steps.default = reader['EXPERIMENT']['save_summary_steps']
        form.experiment.form.throttle.default = reader['EXPERIMENT']['throttle']
        form.experiment.form.validation_batch_size.default = reader['EXPERIMENT']['validation_batch_size']
    if 'NETWORK' in reader.keys():
        form.network.form.hidden_layers.default = reader['NETWORK']['hidden_layers']
        form.network.form.model_name.default = reader['NETWORK']['model_name']
    if 'CUSTOM_MODEL' in reader.keys():
        form.custom_model.form.custom_model_path.default = reader['CUSTOM_MODEL']['custom_model_path']
    if 'TRAINING' in config_reader.read_config(CONFIG_FILE).keys():
        form.training.form.num_epochs.default = reader['TRAINING']['num_epochs']
        form.training.form.batch_size.default = reader['TRAINING']['batch_size']
        form.training.form.optimizer.default = reader['TRAINING']['optimizer']
        form.training.form.learning_rate.default = reader['TRAINING']['learning_rate']
        form.training.form.l1_regularization.default = reader['TRAINING']['l1_regularization']
        form.training.form.l2_regularization.default = reader['TRAINING']['l2_regularization']
        form.training.form.dropout.default = reader['TRAINING']['dropout']
        form.training.form.activation_fn.default = reader['TRAINING']['activation_fn']
    form.network.form.process()
    form.experiment.form.process()
    form.training.form.process()
    form.custom_model.form.process()

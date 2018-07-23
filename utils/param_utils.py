from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm
import pandas as pd
from config import config_reader


def define_param_form(target_type):
    if target_type == 'numerical':
        return GeneralRegressorForm()
    else:
        return GeneralClassifierForm()


def get_number_inputs(categories):
    return len([categories[i] for i in range(len(categories)) if categories[i] != 'none']) - 1


def get_number_outputs(target, data):
    target_type = data.Category[target]
    return 1 if target_type == 'numerical' else data['#Unique Values'][target]


def get_number_samples(file):
    return len(pd.read_csv(file).index)


def get_hidden_layers(INPUT_DIM, OUTUPUT_DIM, num_samples, alpha=2):
    size = num_samples / (alpha * (INPUT_DIM + OUTUPUT_DIM))
    return str(int(round(size)))


def get_defaults_param_form(form, CONFIG_FILE, data, target, train_file):
    number_inputs = get_number_inputs(data.Category)
    number_outputs = get_number_outputs(target, data)
    num_samples = get_number_samples(train_file)

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

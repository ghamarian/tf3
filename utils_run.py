import os
import tensorflow as tf
import math
import socket
from contextlib import closing
import json


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def get_dictionaries(features, categories, fs, target):
    dict_types = {}
    categoricals = {}
    cont = 0
    for k, v in features.items():
        if categories[cont] != 'none':
            dict_types[k] = categories[cont]
            if categories[cont] == 'categorical':
                categoricals[k] = fs.cat_unique_values_dict[k]
        cont += 1
    if target in categoricals.keys():
        categoricals.pop(target)
    return dict_types, categoricals


def get_eval_results(directory, config_writer, CONFIG_FILE):
    results = {}
    if not os.path.isfile(os.path.join(directory, 'export.log')):
        return results

    log_file = json.load(open(os.path.join(directory, 'export.log'), 'r'))

    max_acc = 0
    max_acc_index = 0
    min_loss = math.inf
    min_loss_index = 0
    for k, v in log_file.items():
        step = str(int(v['global_step']))
        if 'accuracy' in v.keys():
            acc = v['accuracy']
            if max_acc < acc:
                max_acc = acc
                max_acc_index = step
        else:
            acc = 'N/A'
        loss = v['average_loss']
        if min_loss > loss:
            min_loss = loss
            min_loss_index = step

        results[k.split('/')[-1]] = {'accuracy': float("{0:.3f}".format(acc)), 'loss': float("{0:.3f}".format(loss)), 'step': step}
    # SAVE best model
    config_writer.add_item('BEST_MODEL', 'max_acc',str(float("{0:.3f}".format(max_acc))))
    config_writer.add_item('BEST_MODEL', 'max_acc_index', str(max_acc_index))
    config_writer.add_item('BEST_MODEL', 'min_loss', str(float("{0:.3f}".format(min_loss))))
    config_writer.add_item('BEST_MODEL', 'min_loss_index', str(min_loss_index))
    config_writer.write_config(CONFIG_FILE)
    return results


def change_model_default(new_model, CONFIG_FILE, config_reader):
    text = 'model_checkpoint_path: "model.ckpt-number"\n'.replace('number', new_model)
    path = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    with open(path + '/checkpoint') as f:
        content = f.readlines()
    content[0] = text
    file = open(path + '/checkpoint', 'w')
    file.write(''.join(content))
    file.close()

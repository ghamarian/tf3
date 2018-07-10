import os
import tensorflow as tf
import math
import socket
from contextlib import closing


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


def get_acc(directory, config_writer, CONFIG_FILE):
    checkpoints = []
    accuras = {}
    eval_dir = os.path.join(directory, 'eval')
    max_acc = 0
    max_acc_index = 0
    min_loss = math.inf
    min_loss_index = 0
    if os.path.exists(eval_dir):
        files_checkpoints = os.listdir(directory)
        for file in files_checkpoints:
            if '.meta' in file:
                # TODO get the last one ?
                checkpoints.append(file.split('.')[1].split('-')[-1])
        path_to_events_file = os.path.join(eval_dir, os.listdir(eval_dir)[-1])
        for e in tf.train.summary_iterator(path_to_events_file):
            if str(e.step) in checkpoints:
                accuras[e.step] = {}
                for v in e.summary.value:
                    if v.tag == 'average_loss':
                        accuras[e.step]['loss'] = float("{0:.3f}".format(v.simple_value))
                        if v.simple_value < min_loss:
                            min_loss = v.simple_value
                            min_loss_index = str(e.step)
                    elif v.tag == 'accuracy':
                        accuras[e.step]['accuracy'] =  float("{0:.3f}".format(v.simple_value))
                        if v.simple_value > max_acc:
                            max_acc = v.simple_value
                            max_acc_index = str(e.step)

    # SAVE best model
    config_writer.add_item('BEST_MODEL', 'max_acc',str(float("{0:.3f}".format(max_acc))))
    config_writer.add_item('BEST_MODEL', 'max_acc_index', str(max_acc_index))
    config_writer.add_item('BEST_MODEL', 'min_loss', str(float("{0:.3f}".format(min_loss))))
    config_writer.add_item('BEST_MODEL', 'min_loss_index', str(min_loss_index))
    config_writer.write_config(CONFIG_FILE)
    return accuras


def change_model_default(new_model, CONFIG_FILE, config_reader):
    text = 'model_checkpoint_path: "model.ckpt-number"\n'.replace('number', new_model)
    path = config_reader.read_config(CONFIG_FILE).all()['checkpoint_dir']
    with open(path + '/checkpoint') as f:
        content = f.readlines()
    content[0] = text
    file = open(path + '/checkpoint', 'w')
    file.write(''.join(content))
    file.close()

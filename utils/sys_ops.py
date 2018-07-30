import os
import socket
from tensorflow.python.platform import gfile
from contextlib import closing
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

def mkdir_recursive(path):
    if not path:
        return
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)


def delete_recursive(paths, export_dir):
    if os.path.isdir(export_dir):
        for p in paths:
            if os.path.exists(os.path.join(export_dir, p)):  gfile.DeleteRecursively(os.path.join(export_dir, p))


def copyfile(src, dst):
    """Copy the contents (no metadata) of the file named src to a file named dst"""
    from shutil import copyfile
    if os.path.exists(src): copyfile(src, dst)


def abs_path_of(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('0.0.0.0', 0))
        return str(s.getsockname()[1])


def save_filename(target, dataset_form_field, dataset_type, dataset_name, sess):
    dataset_form_field.filename = dataset_name + '.csv'
    dataset_file = dataset_form_field
    if dataset_file:
        dataset_filename = secure_filename(dataset_file.filename)
        destination = os.path.join(target, dataset_filename)
        dataset_file.save(destination)
        sess.set(dataset_type, destination)
    return True


def change_checkpoints(config, resume_from):

    rdir = os.path.join(config.get('PATHS', 'export_dir'), resume_from)
    cdir = config.get('PATHS', 'checkpoint_dir')

    for p in Path(cdir).glob("model.*"):
        p.unlink()

    for p in Path(rdir).glob("model.*"):
        shutil.copy(p, os.path.join(cdir, p.name))

    shutil.copy(os.path.join(rdir, 'checkpoint'), os.path.join(cdir, 'checkpoint'))


def delete_configs(config, dataset, username):
    if config != 'all':
        paths = [os.path.join('user_data', username, dataset, config)]
    else:
        paths = [os.path.join('user_data', username, dataset, d) for d in
                 os.listdir(os.path.join('user_data', username, dataset)) if
                 os.path.isdir(os.path.join('user_data', username, dataset, d))]
    for path in paths:
        shutil.rmtree(path)

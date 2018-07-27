import pytest
import os
import socket
from contextlib import closing
from werkzeug.utils import secure_filename
from utils import sys_ops

paths = ['a.ini', 'a2.ini']
export_dir = 'test_recursive/test'


def test_mkdir_recurstive():
    path = os.path.join(export_dir, paths[1])
    sys_ops.mkdir_recursive(path)
    assert os.path.exists(path) == True


def test_delete_recursive():
    sys_ops.delete_recursive(paths, export_dir)
    assert os.path.exists(os.path.join(export_dir, paths[0])) == False
    assert os.path.exists(os.path.join(export_dir, paths[1])) == False


def test_copyfile():
    filename = 'main.py'
    destiny = 'test_recursive/default.ini'
    sys_ops.copyfile(filename, destiny)
    assert os.path.isfile(destiny) == True


def test_not_existing_copyfile():
    filename = 'maindfasdfas.py'
    destiny = 'test_recursive/default2.ini'
    sys_ops.copyfile(filename, destiny)
    assert os.path.isfile(destiny) == False


def test_find_free_port():
    port = sys_ops.find_free_port()
    p = int(port)
    assert isinstance(port, str)
    assert isinstance(p, int)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.bind(("127.0.0.1", p))

    s.close()



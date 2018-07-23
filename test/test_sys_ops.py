import pytest
import os
import socket
from contextlib import closing
from werkzeug.utils import secure_filename

# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4
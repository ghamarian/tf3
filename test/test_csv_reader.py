from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader
import utils
import sys
import pytest
import numpy as np

import tensorflow as tf

tf.enable_eager_execution()

CONFIG_FILE = "config/default.ini"


@pytest.fixture
def train_reader():
    return TrainCSVReader(utils.abs_path_of(CONFIG_FILE))


@pytest.fixture
def validation_reader():
    return ValidationCSVReader(utils.abs_path_of(CONFIG_FILE))


def test_get_label_name(train_reader, validation_reader):
    lable_name = train_reader._get_label_name()
    assert lable_name == 'species'
    columns = train_reader._column_names()
    np.testing.assert_array_equal(columns, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

    lable_name = validation_reader._get_label_name()
    assert lable_name == 'species'
    columns = validation_reader._column_names()
    np.testing.assert_array_equal(columns, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])


def test_dataset(train_reader):
    for x, y in train_reader.make_dataset_from_config({'batch_size': 128}):
        print(y.shape)

def test_feature_columns(train_reader):
    a = train_reader._feature_names().values
    print(a, type(a), a.shape)
    np.testing.assert_array_equal(train_reader._feature_names(), ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

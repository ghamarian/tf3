import utils
import config_reader
import pytest
from collections import OrderedDict

CONFIG_FILE = "config/default.ini"


@pytest.fixture
def config():
    return config_reader.read_config(utils.abs_path_of(CONFIG_FILE))


def test_from_process(config):
    print(config._from_process())

def test_hidden_layers(config):
    a = config.hidden_layers()
    assert len(a) == 2
    assert isinstance(a, list)
    assert a[0] == [32, 16, 16]
    assert a[1] == [16, 8, 4]


def test_training(config):
    a = config.training()
    b = OrderedDict(
        [('num_epochs', '2000'), ('learning_rate', '0.001'), ('batch_size', '32'), ('validation_batch_size', '344'),
         ('optimizer', 'adam'), ('l1_regularization', '0.002'), ('l2_regularization', '0.002'),
         ('dropout_keep_probability', '0.4')])
    assert a == b

import utils
import config_reader
import pytest
from collections import OrderedDict

CONFIG_FILE = "config/default.ini"


@pytest.fixture
def config():
    return config_reader.read_config(utils.abs_path_of(CONFIG_FILE))


def test_from_process(config: config_reader.CustomConfigParser):
    print(config._from_process())


def test_hidden_layers(config: config_reader.CustomConfigParser):
    a = config.hidden_layers()
    assert len(a) == 2
    assert isinstance(a, list)
    assert a[0] == [32, 16, 16]
    assert a[1] == [16, 8, 4]


def test_training(config: config_reader.CustomConfigParser):
    a = config.training()
    print(a)
    # b = dict(
    #     [('num_epochs', '2000'), ('learning_rate', '0.001'), ('batch_size', '32'), ('validation_batch_size', '344'),
    #      ('optimizer', 'Adagrad'), ('l1_regularization', '0.002'), ('l2_regularization', '0.002'),
    #      ('dropout_keep_probability', '0.4')])
    b = {'num_epochs': '2000', 'learning_rate': '0.001', 'batch_size': '32', 'validation_batch_size': '344',
     'optimizer': 'Adagrad', 'l1_regularization': '0.002', 'l2_regularization': '0.002', 'dropout_keep_probability': '0.4'}
    assert a == b



def test_all(config: config_reader.CustomConfigParser):
    all = {'num_epochs': 2000, 'learning_rate': 0.001, 'batch_size': 32, 'validation_batch_size': 344,
           'optimizer': 'Adagrad', 'l1_regularization': 0.002, 'l2_regularization': 0.002, 'dropout_keep_probability': 0.4,
           'experiment_id': 'L1_H26_DO0.4_L10.002_L20.002_B32_LR0.001', 'save_checkpoints_steps': 5000,
           'validation_interval': 10, 'initialize_with_checkpoint': '', 'save_summary_steps': 10,
           'keep_checkpoint_max': 5, 'throttle': 50, 'type': 'classification', 'ground_truth_column': '-1',
           'num_classes': '2', 'weight': '1', 'num_layers': '1', 'layer_size': '26',
           'hidden_layers': [[32, 16, 16], [16, 8, 4]], 'batch_norm': 'True', 'residual': 'False',
           'training_file': 'data/iris.csv', 'validation_file': 'data/iris.csv', 'checkpoint_dir': 'checkpoints/enigma',
           'log_folder': 'log/enigma_Diag', 'model_name': 'DNNClassifier'}
    assert (config.all()) == all

def test_update(config: config_reader.CustomConfigParser):
    a = config.all()
    a.update({'num_epochs': 4000})
    assert a['num_epochs'] == 4000


def test_checkpoint_dir(config: config_reader.CustomConfigParser):
    print(config.checkpoint_dir())

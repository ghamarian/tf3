import configparser
import os
import utils
from typing import Dict
import ast
from collections import OrderedDict

PROCESS = 'PROCESS'

TRAINING = 'TRAINING'


class CustomConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_as_slice(self, *args, **kwargs):
        raw_get = self.get(*args, **kwargs)
        if ":" in raw_get:
            return slice(*map(int, raw_get.split(":")))
        else:
            return int(raw_get)

    def get_rel_path(self, *args, **kwargs):
        raw_get = self.get(*args, **kwargs)
        if not raw_get:
            return ""
        if raw_get.startswith('/'):
            return raw_get

        return utils.abs_path_of(raw_get)

    def _from_training(self, param) :
        return self.get(TRAINING, param)

    def _from_network(self, param):
        return self.get('NETWORK', param)

    def _from_process(self):
        return dict(self.items(TRAINING))

    def training(self) -> Dict[str, str]:
        print(self.items(TRAINING))
        return dict(self.items(TRAINING))

    def process(self) -> Dict[str, str]:
        return dict(self.items(PROCESS))

    def train_batch_size(self) -> int:
        return int(self._from_training('batch_size'))

    def learning_rate(self) -> int:
        return int(self._from_training('learning_rate'))

    def validation_batch(self) -> int:
        return int(self._from_training('validation_batch_size'))

    def optimizer(self) -> str:
        return self._from_training('optimizer')

    def l1_reqularization(self) -> float:
        return float(self._from_training('l1_regularization'))

    def l2_reqularization(self) -> float:
        return float(self._from_training('l2_regularization'))

    def num_epochs(self) -> int:
        return int(self._from_training('num_epochs'))

    def hidden_layers(self):
        return ast.literal_eval(self.get('NETWORK', 'hidden_layers'))



# [PROCESS]
# experiment_ID: L${NETWORK:num_layers}_H${NETWORK:layer_size}_DO${TRAINING:dropout_keep_probability}_L1${TRAINING:l1_regularization}_L2${TRAINING:l2_regularization}_B${TRAINING:batch_size}_LR${TRAINING:learning_rate} #empty means auto name
# checkpoint_every:  5000 # in number of iterations
# validation_interval: 10 # in number of iterations, default if omitted:15
# initialize_with_checkpoint: #checkpoints/enigma/training.ckpt-5000
# val_check_after: 5  # in number of iterations, default if omitted:1000
#


def read_config(path):
    config = CustomConfigParser(inline_comment_prefixes=['#'], interpolation=configparser.ExtendedInterpolation())
    config.read(path)

    return config


def get_task_sections(config):
    return {section_name: config[section_name] for section_name in config.sections() if
            section_name.startswith("TASK")}

# config = read_config("config/default.ini")
# print(config.get_slice("FEATURES","columns"))
# print ([1,2,3][config.get_slice("FEATURES","columns")])

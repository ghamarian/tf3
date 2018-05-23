import tensorflow as tf
import config_reader
from typing import Dict
import pandas as pd
from abc import ABCMeta, abstractmethod


class CSVReader(metaclass=ABCMeta):
    def __init__(self, config_path):
        self.config = config_reader.read_config(config_path)
        self.filename = None
        self.batch_size = None
        self.num_epochs = None
        self.label_name = None

    def make_dataset_from_config(self, params: Dict[str, object] = None) -> tf.data.Dataset:
        self._set_params(params)
        return self._make_csv_dataset()

    @abstractmethod
    def _set_params(self, params: Dict[str, object]) -> None:
        pass

    def _get_int_from_config(self, section: str, key: str) -> int:
        return int(self.config[section][key])

    def _get_from_config(self, section: str, key: str) -> str:
        return self.config[section][key]

    def _make_csv_dataset(self):
        return tf.contrib.data.make_csv_dataset([self.filename], self.batch_size, num_epochs=self.num_epochs,
                                                 label_name=self.label_name)
    def _column_names(self) -> pd.DataFrame:
        return pd.read_csv(self.filename, nrows=2).columns

    def _feature_names(self):
        feature_slice = self.config.get_as_slice('FEATURES', 'columns')
        return self._column_names()[feature_slice]

    def _get_label_name(self):
        columns = self._column_names()
        mask = self.config.get_as_slice('TASK0', 'ground_truth_column')
        return columns[mask]

    def get_label_unique_values(self):
        label_column = self._get_label_name()
        df = pd.read_csv(self.filename, usecols=[label_column])
        return df[label_column].unique()

    def _get_param_with_config_default(self, params, section, param_name):
        return params.get(param_name, self._get_int_from_config(section, param_name))

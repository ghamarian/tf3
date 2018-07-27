from feature_selection import FeatureSelection
import pandas as pd
from config.config_writer import ConfigWriter
from utils import run_utils
from config import config_reader

df = pd.read_csv('data_test/iris.csv')
fs = FeatureSelection(df)
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
directory = 'data_test/best_exporter_test'
config_test_file = 'data_test/iris_config_test.ini'
checkpoints = {'1532434574': {'accuracy': 1.0, 'loss': 0.153, 'step': '42'}}
dict_types = {'sepal_length': 'numerical', 'sepal_width': 'numerical', 'petal_length': 'numerical',
              'petal_width': 'numerical', 'class': 'categorical'}
sfeatures = {'sepal_length': '5.8', 'sepal_width': '3.0', 'petal_length': '4.35', 'petal_width': '1.3'}
categories = ['numerical', 'numerical', 'numerical', 'numerical', 'categorical']


def test_get_html_types():
    dict_html = {'sepal_length': 'number', 'sepal_width': 'number', 'petal_length': 'number', 'petal_width': 'number',
                 'class': 'text'}
    assert run_utils.get_html_types(dict_types) == dict_html


def test_get_dictionaries():
    target = 'class'
    features = {'sepal_length': '5.8', 'sepal_width': '3', 'petal_length': '4.35', 'petal_width': '1.3', 'class': 'Iris-setosa'}
    types, cats = run_utils.get_dictionaries(features, categories, fs, target)
    assert types == dict_types
    assert cats == {}

def test_get_eva_results():
    writer = ConfigWriter()
    writer.append_config(config_test_file)
    results = run_utils.get_eval_results(directory, writer, config_test_file)
    assert results == checkpoints
    reader = config_reader.read_config(config_test_file)
    assert reader['BEST_MODEL']['max_acc'] == '1.0'
    assert reader['BEST_MODEL']['max_acc_index'] == '42'
    assert reader['BEST_MODEL']['min_loss'] == '0.153'
    assert reader['BEST_MODEL']['min_loss_index'] == '42'

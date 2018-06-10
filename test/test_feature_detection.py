import pandas as pd
from pprint import pprint
import pytest

from feature_selection import FeatureSelection

numerical_col = ['sepal_length',
                 'sepal_width',
                 'petal_length',
                 'petal_width']

categorical_col = ['species']
bool_col = ['bool_col']
int_col = ['int_col', 'int_big_variance']
hash_col = ['hash_col']

unique_values = {'species': ['setosa', 'versicolor', 'virginica']}

features_datatypes = ['numerical', 'numerical', 'numerical', 'numerical', 'categorical', 'range', 'categorical', 'hash',
                      'hash']


@pytest.fixture
def df():
    return pd.read_csv('data/test.csv')


@pytest.fixture
def fs(df):
    return FeatureSelection(df)


def test_populate_features(df, fs):
    assert numerical_col == fs.numerical_columns
    assert categorical_col == fs.categorical_columns
    assert hash_col == fs.hash_columns
    assert bool_col == fs.bool_columns
    assert int_col == fs.int_columns

    # self.cat_unique_values_dict = {}
    # self.hash_bucket_sizes_dict = {}
    # self.categorical_columns = []
    # self.hash_columns = []
    hash_bucket_sizes = {'hash_col': 10}

    assert unique_values == fs.cat_unique_values_dict
    assert fs.unique_value_size_dict == {'bool_col': 2,
                                         'hash_col': 10,
                                         'int_big_variance': 138,
                                         'int_col': 5,
                                         'species': 3}
    pprint(fs.feature_dict())


def test_group_by(fs):
    assert fs.group_by(features_datatypes) == {'categorical': ['species', 'bool_col'],
                                               'hash': ['int_big_variance', 'hash_col'],
                                               'numerical': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                               'range': ['int_col']}

def test_select_target(fs):

    datatypes = ['numerical', 'numerical', 'numerical', 'numerical', 'categorical', 'range', 'categorical', 'hash', 'hash']
    fs.create_tf_features(datatypes)
    feature_len  = len(fs.feature_columns)
    target = fs.select_target('species')
    assert target.key == 'species'
    assert feature_len - 1 == len(fs.feature_columns)

def test_defaults(fs):
    fs.populate_defaults()

    # pprint(fs.means)
    pprint(fs.modes)
    pprint(fs.frequent_values)
    # pprint(fs.defaults)



    # def populate_defaults(self):
    #     self.means = self.df.mean().to_dict()
    #     self.modes = self.df.mode().iloc[0,:].to_dict()
    #     frequent_values = {}
    #     for col in self.df.columns:
    #         frequent_values.update(self.df[col].value_counts.head(1).to_dict())
    #
    #     self.frequent_values = frequent_values
    #
    #     self.defaults = self.means
    #     self.defaults.update(self.modes)

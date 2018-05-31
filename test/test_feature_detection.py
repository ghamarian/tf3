import pandas as pd
from pprint import pprint

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


def test_populate_features():
    df = pd.read_csv('data/test.csv')
    fs = FeatureSelection(df)
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

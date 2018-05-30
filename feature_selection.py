import pandas as pd
import tensorflow as tf
import itertools
import utils
import lazy_property
from pprint import pprint


class FeatureSelection():
    MAX_CATEGORICAL_SIZE = 5

    def __init__(self, df):
        self.features = {}
        self.df = df
        self.numerical_columns = self.select_columns_with_type('floating')

        self.int_columns = self.select_columns_with_type('integer')
        self.unique_value_size_dict = {key: self.df[key].unique().shape[0] for key in self.int_columns}

        self.bool_columns = self.select_columns_with_type('bool')
        self.unique_value_size_dict.update(dict(itertools.product(self.bool_columns, [2])))

        known_columns = list(
            itertools.chain.from_iterable([self.numerical_columns, self.int_columns, self.bool_columns]))
        self.cat_or_hash_columns = [col_name for col_name in df.columns if col_name not in known_columns]
        self.populate_hash_and_categorical()

        self.column_list = {'numerical': self.numerical_columns,
                            'bool': self.bool_columns,
                            'categorical': self.categorical_columns,
                            'int': self.int_columns}

    def feature_list(self):
        return dict(itertools.chain.from_iterable(
            [itertools.product(self.column_list[key], [key]) for key in self.column_list]))

    def populate_hash_and_categorical(self):
        self.cat_unique_values_dict = {}
        self.categorical_columns = []
        self.hash_columns = []

        for col in self.cat_or_hash_columns:
            unique = self.df[col].unique().tolist()
            if (len(unique) < self.MAX_CATEGORICAL_SIZE):
                self.cat_unique_values_dict[col] = unique
                self.categorical_columns.append(col)
            else:
                self.hash_columns.append(col)
            self.unique_value_size_dict[col] = len(unique)

    def create_tf_features(self, feature_types):
        numerical_features = [tf.feature_column.numeric_column(key) for key in feature_types['numerical']]
        range_features = [tf.feature_column.categorical_column_with_identity(key, self.unique_value_size_dict[key]) for
                          key in feature_types['range']]

        categorical_features = []
        for feature in feature_types['categorical']:
            vocab_list = self.cat_unique_values_dict.get(feature, self.df[feature].unique().tolist())
            categorical_features.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab_list))

        hash_features = [tf.feature_column.categorical_column_with_hash_bucket(key, self.unique_value_size_dict[key])
                         for key in feature_types['hash']]

        self.feature_columns = itertools.chain.from_iterable(
            [numerical_features, categorical_features, hash_features, range_features])

    def select_columns_with_type(self, dftype):
        return self.df.select_dtypes(include=[dftype]).columns.tolist()

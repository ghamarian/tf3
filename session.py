from config.config_writer import ConfigWriter
from config import config_reader
from flask import session, redirect, url_for
from feature_selection import FeatureSelection
import itertools
from utils import feature_util, preprocessing

import pandas as pd

SAMPLE_DATA_SIZE = 5


class Session:
    def __init__(self, app):
        self._config_writer = {}
        self._config = {}
        # self._processes = {}
        # self._ports = {}
        self._app = app

    def add_user(self, user):
        self._config[user] = {}
        self._config_writer[user] = ConfigWriter()

    def reset_user(self):
        user = self.get_session('user')
        self._config[user] = {}
        self._config_writer[user] = ConfigWriter()

    def get_session(self, user_id):
        with self._app.app_context():
            if user_id not in session:
                return redirect(url_for('login'))
            return session[user_id]

    def get_config(self):
        user = self.get_session('user')
        return self._config[user]

    def get_writer(self):
        user = self.get_session('user')
        return self._config_writer[user]

    def get(self, key):
        return self.get_config()[key]

    def set(self, key, value):
        user = self.get_session('user')
        self._config[user][key] = value

    def update_writer_conf(self, conf):
        user = self.get_session('user')
        self._config_writer[user].config = conf

    def update_split(self, train_file, validation_file):
        self.set('train_file', train_file)
        self.set('validation_file', validation_file)
        self.get_writer().add_item('PATHS', 'train_file', train_file)
        self.get_writer().add_item('PATHS', 'validation_file', validation_file)
        self.get_writer().add_item('SPLIT_DF', 'split_df', self.get('split_df'))

    def set_target(self, target):
        self.get_writer().add_item('TARGET', 'target', target)
        self.set('features', self.get('fs').create_tf_features(self.get('category_list'), target))
        self.set('target', target)
        target_type = self.get('data').Category[self.get('target')]
        if target_type == 'range':
            new_categ_list = []
            for categ, feature in zip(self.get('category_list'), self.get('df').columns):
                new_categ_list.append(categ if feature != target else 'categorical')
            self.set('category_list', new_categ_list)
            self.get('data').Category = self.get('category_list')
            self.get('fs').update(self.get('category_list'),
                                  dict(zip(self.get('data').index.tolist(), self.get('data').Defaults)))

    def load_config(self):
        # read saved config
        conf = config_reader.read_config(self.get('config_file'))
        # update files and df in config dict
        self.set('file', conf['PATHS']['file'])
        self.set('train_file', conf['PATHS']['train_file'])
        self.set('validation_file', conf['PATHS']['validation_file'])
        self.set('df', pd.read_csv(conf['PATHS']['file']))
        self.load_features()
        # target select
        target = conf['TARGET']['target']
        self.set_target(target)
        self.update_new_features(list(dict(conf.items('COLUMN_CATEGORIES')).values()), list(self.get('defaults').values()))
        self.update_writer_conf(conf)

    def assign_category(self, df):
        fs = FeatureSelection(df)
        self.set('fs', fs)
        category_list, unique_values, default_list, frequent_values2frequency = fs.assign_category(
            self.get('config_file'), df)
        return category_list, unique_values, default_list, frequent_values2frequency

    def update_new_features(self, cat_columns, default_values):
        self.set('category_list', cat_columns)
        self.get('data').Category = self.get('category_list')
        self.get('data').Defaults = default_values
        self.set('defaults', dict(zip(self.get('data').index.tolist(), default_values)))
        self.get('fs').update(self.get('category_list'), dict(zip(self.get('data').index.tolist(), default_values)))

    def load_features(self):
        # retrieve values from config, assign_category does this
        self.set('df', pd.read_csv(self.get('file')))
        df = self.get('df')
        df.reset_index(inplace=True, drop=True)
        categories, unique_values, default_list, frequent_values2frequency = self.assign_category(df)
        default_values = [str(v) for v in default_list.values()]
        self.set('data',
                 preprocessing.insert_data(df, categories, unique_values, default_list, frequent_values2frequency,
                                          SAMPLE_DATA_SIZE))
        self.set('defaults', dict(zip(self.get('data').index.tolist(), default_values)))
        self.set('category_list', categories)
        return categories

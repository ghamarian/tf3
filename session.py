from config.config_writer import ConfigWriter
from config import config_reader
from flask import session, redirect, url_for
from feature_selection import FeatureSelection
import itertools

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


    # def update_config(self, key, value):
    #     config = self.get_config()
    #     config[key] = value

    def load_config(self):
        # read saved config
        CONFIG_FILE = self.get('config_file')

        conf = config_reader.read_config(CONFIG_FILE)

        # update files and df in config dict
        self.set('file', conf['PATHS']['file'])
        self.set('train_file', conf['PATHS']['train_file'])
        self.set('validation_file', conf['PATHS']['validation_file'])

        self.set('df', pd.read_csv(conf['PATHS']['file']))

        # retrieve values from config, assign_category does this
        df = self.get('df')
        df.reset_index(inplace=True, drop=True)
        categories, unique_values, default_list, frequent_values2frequency = self.assign_category(df)
        default_values = [str(v) for v in default_list.values()]

        data = df.head(SAMPLE_DATA_SIZE).T
        data.insert(0, 'Defaults', default_list.values())
        data.insert(0, '(most frequent, frequency)', frequent_values2frequency.values())
        data.insert(0, 'Unique Values', unique_values)
        data.insert(0, 'Category', categories)

        sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
        data.columns = list(
            itertools.chain(['Category', '#Unique Values', '(Most frequent, Frequency)', 'Defaults'],
                            sample_column_names))

        self.set('data', data)
        self.set('defaults', dict(zip(self.get('data').index.tolist(), default_values)))
        self.set('category_list', categories)

        # target select
        target = conf['TARGET']['target']
        self.set('features', self.get('fs').create_tf_features(self.get('category_list'), target))
        self.set('target', target)
        target_type = self.get('data').Category[self.get('target')]
        if 'range' in target_type:
            new_categ_list = []
            for categ, feature in zip(self.get('category_list'), self.get('df').columns):
                new_categ_list.append(categ if feature != target else 'categorical')
            self.set('category_list', new_categ_list)
            self.get('data').Category = self.get('category_list')
            self.get('fs').update(self.get('category_list'),
                                  dict(zip(self.get('data').index.tolist(), self.get('data').Defaults)))

        self.update_writer_conf(conf)

    def assign_category(self, df):
        fs = FeatureSelection(df)
        self.set('fs', fs)
        feature_dict = fs.feature_dict()
        unique_values = [fs.unique_value_size_dict.get(key, -1) for key in df.columns]
        category_list = [feature_dict[key] for key in df.columns]
        CONFIG_FILE = self.get('config_file')
        if 'COLUMN_CATEGORIES' in config_reader.read_config(CONFIG_FILE).keys():
            category_list = []
            for key in df.columns:
                category_list.append(config_reader.read_config(CONFIG_FILE)['COLUMN_CATEGORIES'][key])
        default_list = fs.defaults
        frequent_values2frequency = fs.frequent_values2frequency
        return category_list, unique_values, default_list, frequent_values2frequency




import tensorflow as tf
import utils

from train_csv_reader import TrainCSVReader

tf.logging.set_verbosity(tf.logging.DEBUG)

CONFIG_FILE = "config/default.ini"
csv_reader = TrainCSVReader(utils.abs_path_of(CONFIG_FILE))

feature_columns = [
    tf.feature_column.numeric_column(key) for key in csv_reader.feature_names()
]

model = tf.estimator.DNNClassifier([10, 5, 5], feature_columns, n_classes=3, model_dir='checkpoint',
                                      label_vocabulary=['setosa', 'versicolor', 'virginica'])


model.train(lambda: csv_reader.make_dataset_from_config({'batch_size': 32}))


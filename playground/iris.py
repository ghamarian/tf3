import tensorflow as tf
import csv

BATCH_SIZE = 32
NUM_EPOCHS = 1000

tf.logging.set_verbosity(tf.logging.DEBUG)

with open('data/iris.csv', 'r') as f:
    csv_columns = csv.reader(f, delimiter=',')
    feature_names = next(csv_columns)

feature_list = [tf.feature_column.numeric_column(feature) for feature in feature_names[:-1]]


def train_input_fn():
    return tf.contrib.data.make_csv_dataset('data/iris.csv', BATCH_SIZE, num_epochs=NUM_EPOCHS, label_name='species')


tf.estimator.DNNClassifier([3], model_dir='amir', feature_columns=feature_list, n_classes=3,
                           label_vocabulary=['setosa', 'versicolor', 'virginica']).train(train_input_fn)


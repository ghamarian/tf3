import tensorflow as tf
from tensorflow.contrib.estimator import multi_class_head
from tensorflow.python.estimator.estimator import Estimator
import csv
import pandas as pd

BATCH_SIZE = 32
NUM_EPOCHS = 1000

tf.logging.set_verbosity(tf.logging.DEBUG)

with open('iris.csv', 'r') as f:
    csv_columns = csv.reader(f, delimiter=',')
    feature_names = next(csv_columns)

feature_list = [tf.feature_column.numeric_column(feature) for feature in feature_names[:-1]]


# feature_list += tf.feature_column.categorical_column_with_vocabulary_list( 'species', ['setosa', 'versicolor', 'virginica'] )

def my_model_fn(features, labels, mode):
    head = multi_class_head(3)
    net = tf.feature_column.input_layer(features, feature_list)
    h1 = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)(net)
    logits = tf.keras.layers.Dense(3)(h1)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    return head.create_estimator_spec(features, mode, logits=logits, labels=labels, optimizer=optimizer)


def train_input_fn():
    return tf.contrib.data.make_csv_dataset('iris-new.csv', BATCH_SIZE, num_epochs=NUM_EPOCHS, label_name='species')

def pandas_train_input_fn(df):
    result = df['species']
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=df['species'],
        batch_size=32,
        shuffle=True,
        queue_capacity=1000,
        num_epochs=10
    )


estimator = Estimator(my_model_fn)

# estimator.train(train_input_fn)
estimator.train(pandas_train_input_fn(pd.read_csv('iris.csv')))

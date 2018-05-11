import tensorflow as tf
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.DEBUG)


feature_columns = [
    tf.feature_column.numeric_column("size"),
    tf.feature_column.categorical_column_with_vocabulary_list("type", ["apt", "house"])
]

def train_input_fn():
    features = {"size": [1000, 2000, 3000, 1000, 2000, 3000],
                "type": ["house", "house", "house", "apt", "apt", "apt"]}
    labels = [500, 1000, 1500, 700, 1300, 1900]

    return features, labels


def numpy_train_input_fn(sqft, prop_type, price):
    return tf.estimator.inputs.numpy_input_fn(
        x = {"size": sqft, "type": prop_type},
        y = price,
        batch_size=128,
        num_epochs=10,
        shuffle=True,
        queue_capacity=1000
    )

def pandas_train_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df['price'],
        batch_size=128,
        shuffle=True,
        queue_capacity=1000,
        num_epochs=10
    )


model = tf.estimator.LinearRegressor(feature_columns, "checkpoint")

# model.train(train_input_fn, steps=1000)
# steps is for mini-batches (training step)
model.train(pandas_train_input_fn(pd.read_csv("input.csv")), steps=100)



import tensorflow as tf
import pandas as pd

feature_columns = [
    tf.feature_column.numeric_column("size"),
    tf.feature_column.categorical_column_with_vocabulary_list("type", ["house", "apt"])
]


def train_input_fn():
    features = {
        "size": [1000, 2000, 3000, 1000, 2000, 3000],
        "type": ["house", "house", "house", "apt", "apt", "apt"]
    }
    lables = [500, 1000, 1500, 700, 1300, 1900]

    return features, lables


def pandas_train_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=df['price'],
        batch_size=128,
        shuffle=True,
        queue_capacity=1000,
        num_epochs=10
    )


model = tf.estimator.DNNRegressor(3, feature_columns, "checkpoints")
model.train()

import tensorflow as tf

# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.DEBUG)

BATCH_SIZE = 32
NUM_EPOCHS = 100


def train_input_fn():
    return tf.contrib.data.make_csv_dataset('input.csv', BATCH_SIZE, num_epochs=NUM_EPOCHS, label_name='price')


feature_columns = [
    tf.feature_column.numeric_column("size"),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("type", ["house", "apt"]))
]

model = tf.estimator.DNNRegressor([3], feature_columns, model_dir="checkpoint")

model.train(train_input_fn)



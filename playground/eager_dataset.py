import tensorflow as tf
import functools

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.DEBUG)


dataset = tf.contrib.data.make_csv_dataset("data/test.csv", 2, num_epochs=10, label_name="species")


def parser(record, second, key):
    record.update({key: tf.as_string(record[key])})
    return record, second

# lambda x: tf.cast(x['int_big_variance'], tf.string)
dataset = dataset.map(functools.partial(parser, key='int_big_variance'))


for batch in dataset:
    print(batch)


# def train_input_fn():
#     return tf.contrib.data._make_csv_dataset("data/iris.csv", 64, num_epochs=10, label_name="price")

# a = dataset.make_one_shot_iterator().get_next()


# feature_columns = [
#     tf.feature_column.numeric_column('size'),
#     tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('type', ['house', 'apt']))
# ]
#
# tf.estimator.DNNRegressor([3], feature_columns=feature_columns).train(train_input_fn)

# with tf.Session() as sess:
#     for i in range(10):
#         print(sess.run(a))
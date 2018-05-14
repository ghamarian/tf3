import tensorflow as tf

# tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.DEBUG)

dataset = tf.contrib.data.make_csv_dataset("input.csv", 2, num_epochs=10, label_name="price")

def train_input_fn():
    return tf.contrib.data.make_csv_dataset("input.csv", 64, num_epochs=10, label_name="price")

a = dataset.make_one_shot_iterator().get_next()

# for batch in dataset:
#     print(batch)

feature_columns = [
    tf.feature_column.numeric_column('size'),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('type', ['house', 'apt']))
]

tf.estimator.DNNRegressor([3], feature_columns=feature_columns).train(train_input_fn)

# with tf.Session() as sess:
#     for i in range(10):
#         print(sess.run(a))
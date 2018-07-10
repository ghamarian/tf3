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
    return tf.contrib.data.make_csv_dataset('data/iris-train.csv', BATCH_SIZE, num_epochs=NUM_EPOCHS, label_name='class')


def validation_input_fn():
    return tf.contrib.data.make_csv_dataset('data/iris-validation.csv', BATCH_SIZE, num_epochs=NUM_EPOCHS, label_name='class')


model = tf.estimator.DNNClassifier([3], model_dir='amir', feature_columns=feature_list, n_classes=3,activation_fn=tf.nn.relu,
    dropout=None, label_vocabulary=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']).train(train_input_fn)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=5000)
eval_spec= tf.estimator.EvalSpec(input_fn=validation_input_fn,
            steps=None,  # How many batches of test data
            start_delay_secs=0, throttle_secs=1)
tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
import tensorflow as tf
import utils
from tensorflow.python import debug as tf_debug
import shutil

from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader

shutil.rmtree("checkpoint", ignore_errors=True)
tf.reset_default_graph()

tf.logging.set_verbosity(tf.logging.DEBUG)

CONFIG_FILE = "config/default.ini"
train_reader = TrainCSVReader(utils.abs_path_of(CONFIG_FILE))

feature_columns = [
    tf.feature_column.numeric_column(key) for key in train_reader._feature_names()
]

runConfig = tf.estimator.RunConfig(model_dir="checkpoint",
                                   save_checkpoints_steps=100, # this sets the minimum for evaluation. you can only evaluate based on time
                                   save_summary_steps=100,
                                   keep_checkpoint_max=5)

model = tf.estimator.DNNClassifier([10, 5, 5], feature_columns, n_classes=3,
                                   label_vocabulary=['setosa', 'versicolor', 'virginica'], config=runConfig)

# hook = tf_debug.TensorBoardDebugHook("Amirs-MacBook-Pro-2.local:7000")

# model.train(lambda: csv_reader.make_dataset_from_config({'batch_size': 32}), hooks=[hook])
# model.train(lambda: csv_reader.make_dataset_from_config({'batch_size': 32}))

validate_reader = ValidationCSVReader(utils.abs_path_of(CONFIG_FILE))
# model.evaluate(lambda: validate_csv_reader.make_dataset_from_config({}))

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_reader.make_dataset_from_config({'batch_size': 32}), max_steps=5000)

# TODO what are all these aparameters
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: validate_reader.make_dataset_from_config({}),
    steps=1,  # How many batches of test data
    start_delay_secs=0, throttle_secs=1)
# throttle_secs=1)

# start evaluating after N seconds
# ,throttle_secs = 5)
# evaluate every N seconds
# , exporters = exporter)

tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

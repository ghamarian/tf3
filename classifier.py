import tensorflow as tf
import utils
from tensorflow.python import debug as tf_debug
import shutil
import config_reader
from model_builder import ModelBuilder

from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader

HIDDEN_LAYERS = 'hidden_layers'

MAX_STEPS = 'max_steps'

KEEP_CHECKPOINT_MAX = 'keep_checkpoint_max'

SAVE_SUMMARY_STEPS = 'save_summary_steps'

SAVE_CHECKPOINTS_STEPS = 'save_checkpoints_steps'


class Classifier:

    def __init__(self, params, train_csv_reader, validation_csv_reader):
        self.params = params
        self.checkpoint_dir = params['checkpoint_dir']
        self.train_csv_reader = train_csv_reader
        self.validation_csv_reader = validation_csv_reader
        self.feature_columns = [tf.feature_column.numeric_column(key) for key in self.train_csv_reader._feature_names()]

        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.reset_default_graph()

        self._create_run_config()
        self._create_model()
        self._create_specs()

    def clear_checkpoint(self):
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def run(self):
        tf.estimator.train_and_evaluate(self.model, self.train_spec, self.eval_spec)

    def _train_input_fn(self):
        return self.train_csv_reader.make_dataset_from_config(self.params)

    def _validation_input_fn(self):
        return self.validation_csv_reader.make_dataset_from_config(self.params)

    def _create_model(self):
        # TODO hidden_layers to be fixed
        hidden_layers = self.params[HIDDEN_LAYERS][0]
        label_vocabulary = self.train_csv_reader.label_unique_values()

        mb = ModelBuilder(self.feature_columns)
        # b = a.grab("tensorflow.python.estimator.canned.linear.LinearRegressor", self.feature_columns)

        self.params['n_classes'] = len(label_vocabulary)
        self.params['label_vocabulary'] = label_vocabulary.tolist()
        self.params['config'] = self.runConfig
        self.params['hidden_units'] = hidden_layers

        self.model = mb.create_from_model_name(self.params['model_name'], self.feature_columns, self.params)

        # self.model = tf.estimator.DNNClassifier(hidden_layers, self.feature_columns, n_classes=len(label_vocabulary), label_vocabulary=label_vocabulary.tolist(), config=self.runConfig)

    def _create_specs(self):
        max_steps = self.params[MAX_STEPS]
        self.train_spec = tf.estimator.TrainSpec(
            input_fn=self._train_input_fn, max_steps=max_steps)

        # TODO throttle and start_delay and steps?
        self.eval_spec = tf.estimator.EvalSpec(
            input_fn=self._validation_input_fn,
            steps=1,  # How many batches of test data
            start_delay_secs=0, throttle_secs=1)

    def _create_run_config(self):
        save_checkpoints_steps = self.params[SAVE_CHECKPOINTS_STEPS]
        save_summary_steps = self.params[SAVE_SUMMARY_STEPS]
        keep_checkpoint_max = self.params[KEEP_CHECKPOINT_MAX]
        self.runConfig = tf.estimator.RunConfig(model_dir=self.checkpoint_dir,
                                                save_checkpoints_steps=save_checkpoints_steps,
                                                save_summary_steps=save_summary_steps,
                                                keep_checkpoint_max=keep_checkpoint_max)


CONFIG_FILE = "config/default.ini"
config = config_reader.read_config(CONFIG_FILE)
train_csv_reader = TrainCSVReader(config)
validation_csv_reader = ValidationCSVReader(config)

params = {'batch_size': 32,
          'max_steps': 5000,
          'save_checkpoints_steps': 100,
          'save_summary_steps': 100,
          'keep_checkpoint_max': 5,
          'num_epochs': 200,
          'validation_batch_size': 300
          }
config_params = config.all()
config_params.update(params)

classifier = Classifier(config_params, train_csv_reader, validation_csv_reader)
classifier.clear_checkpoint()
classifier.run()

# hook = tf_debug.TensorBoardDebugHook("localhost:7000")

# model.train(lambda: csv_reader.make_dataset_from_config({'batch_size': 32}), hooks=[hook])
# model.train(lambda: csv_reader.make_dataset_from_config({'batch_size': 32}))

# model.evaluate(lambda: validate_csv_reader.make_dataset_from_config({}))


# throttle_secs=1)

# start evaluating after N seconds
# ,throttle_secs = 5)
# evaluate every N seconds
# , exporters = exporter)

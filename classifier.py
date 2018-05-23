import tensorflow as tf
import utils
from tensorflow.python import debug as tf_debug
import shutil

from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader


class Classifier():

    def __init__(self, config_file, params):
        self.config_file = utils.abs_path_of(config_file)
        self.params = params
        self.train_csv_reader = TrainCSVReader(self.config_file)
        self.feature_columns = [tf.feature_column.numeric_column(key) for key in self.train_csv_reader._feature_names()]

        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.reset_default_graph()

        self._create_run_config()
        self._create_model()
        self._create_specs()

    def clear_checkpoint(self):
        shutil.rmtree("checkpoint", ignore_errors=True)

    def run(self):
        tf.estimator.train_and_evaluate(self.model, self.train_spec, self.eval_spec)

    def _train_input_fn(self):
        return self.train_csv_reader.make_dataset_from_config(self.params)

    def _validation_input_fn(self):
        return ValidationCSVReader(self.config_file).make_dataset_from_config(self.params)

    def _create_model(self):
        self.model = tf.estimator.DNNClassifier([10, 5, 5], self.feature_columns, n_classes=3,
                                                label_vocabulary=['setosa', 'versicolor', 'virginica'],
                                                config=self.runConfig)

    def _create_specs(self):
        max_steps = self.params['max_steps']
        self.train_spec = tf.estimator.TrainSpec(
            input_fn=self._train_input_fn, max_steps=max_steps)

        self.eval_spec = tf.estimator.EvalSpec(
            input_fn=self._validation_input_fn,
            steps=1,  # How many batches of test data
            start_delay_secs=0, throttle_secs=1)

    def _create_run_config(self):
        save_checkpoints_steps = self.params['save_checkpoints_steps']
        save_summary_steps = self.params['save_summary_steps']
        keep_checkpoint_max = self.params['keep_checkpoint_max']

        self.runConfig = tf.estimator.RunConfig(model_dir="checkpoint",
                                                save_checkpoints_steps=save_checkpoints_steps,
                                                save_summary_steps=save_summary_steps,
                                                keep_checkpoint_max=keep_checkpoint_max)


CONFIG_FILE = "config/default.ini"
classifier = Classifier(CONFIG_FILE,
                        {'batch_size': 32,
                         'max_steps': 5000,
                         'save_checkpoints_steps': 100,
                         'save_summary_steps': 100,
                         'keep_checkpoint_max': 5
                         })
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

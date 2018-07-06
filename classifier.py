import tensorflow as tf
import shutil
import numpy as np
from config import config_reader
from model_builder import ModelBuilder
from keras.models import load_model
from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader

HIDDEN_LAYERS = 'hidden_layers'

MAX_STEPS = 'max_steps'

KEEP_CHECKPOINT_MAX = 'keep_checkpoint_max'

SAVE_SUMMARY_STEPS = 'save_summary_steps'

SAVE_CHECKPOINTS_STEPS = 'save_checkpoints_steps'


class Classifier:

    def __init__(self, params, train_csv_reader, validation_csv_reader, feature_columns, label_unique_values):
        self.params = params
        self.checkpoint_dir = params['checkpoint_dir']
        self.train_csv_reader = train_csv_reader
        self.validation_csv_reader = validation_csv_reader
        self.feature_columns = feature_columns
        self.label_unique_values = label_unique_values

        # self.feature_columns = [tf.feature_column.numeric_column(key) for key in self.train_csv_reader._feature_names()]

        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.reset_default_graph()

        self._create_run_config()
        self._create_model()
        self._create_specs()

    def clear_checkpoint(self):
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def run(self):
        tf.estimator.train_and_evaluate(self.model, self.train_spec, self.eval_spec)

    def predict(self, features, target, df):
        del features[target]
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=self.input_predict_fn(features, df),
                                                              y=None, num_epochs=1, shuffle=False)
        predictions = list(self.model.predict(input_fn=predict_input_fn))
        print(predictions)
        if 'predictions' in predictions[0].keys():
            return predictions[0]['predictions'][0]
        return predictions[0]['classes'][0].decode("utf-8")

    def input_predict_fn(self, features,  df):
        input_predict = {}
        for k, v in features.items():
            input_predict[k] = np.array([v]).astype(df[k].dtype)
        # input_predict.pop(target, None)
        return input_predict

    def _train_input_fn(self):
        return self.train_csv_reader.make_dataset_from_config(self.params)

    def _validation_input_fn(self):
        return self.validation_csv_reader.make_dataset_from_config(self.params)

    def _create_model(self):
        # TODO hidden_layers to be fixed
        # hidden_layers = self.params[HIDDEN_LAYERS][0]
        hidden_layers = self.params[HIDDEN_LAYERS]

        mb = ModelBuilder()
        # b = a.grab("tensorflow.python.estimator.canned.linear.LinearRegressor", self.feature_columns)

        self.params['n_classes'] = len(self.label_unique_values) if self.label_unique_values is not None else 0
        self.params['label_vocabulary'] = self.label_unique_values
        self.params['config'] = self.runConfig
        self.params['hidden_units'] = hidden_layers
        self.params['dnn_hidden_units'] = hidden_layers
        self.params['dnn_dropout'] = self.params['dropout']
        self.params['dnn_optimizer'] = self.params['optimizer']
        self.params['linear_optimizer'] = self.params['optimizer']
        self.params['activation_fn'] = getattr(tf.nn, self.params['activation_fn'])

        self.model = mb.create_from_model_name(self.params['model_name'], self.feature_columns, self.params)

    def _create_specs(self):
        max_steps = self.params[MAX_STEPS]
        self.train_spec = tf.estimator.TrainSpec(
            input_fn=self._train_input_fn, max_steps=max_steps)

        # TODO throttle and start_delay and steps?
        # summary_hook = tf.train.SummarySaverHook(
        #     save_steps=1,
        #     output_dir='./tmp/rnnStats',
        #     scaffold=tf.train.Scaffold(),
        #     summary_op=tf.summary.merge_all())
        #

        self.eval_spec = tf.estimator.EvalSpec(
            input_fn=self._validation_input_fn,
            steps=None,  # How many batches of test data
            start_delay_secs=0, throttle_secs=1)

    def _create_run_config(self):
        save_checkpoints_steps = self.params[SAVE_CHECKPOINTS_STEPS]
        save_summary_steps = self.params[SAVE_SUMMARY_STEPS]
        keep_checkpoint_max = self.params[KEEP_CHECKPOINT_MAX]
        self.runConfig = tf.estimator.RunConfig(model_dir=self.checkpoint_dir,
                                                save_checkpoints_steps=save_checkpoints_steps,
                                                save_summary_steps=save_summary_steps,
                                                keep_checkpoint_max=keep_checkpoint_max)


class KerasClassifier:

    def __init__(self,params, train_csv_reader, validation_csv_reader, feature_columns, label_unique_values):
       self.params = params
       self.checkpoint_dir = params['checkpoint_dir']
       self.train_csv_reader = train_csv_reader
       self.validation_csv_reader = validation_csv_reader

       tf.logging.set_verbosity(tf.logging.DEBUG)

       tf.reset_default_graph()

       self._create_run_config()
       self._create_model()
       self._create_specs()

    def predict(self, features, target, df):
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={self.input_name: self.input_predict_fn(features, target, df)},
                                                               y=None, num_epochs=1, shuffle=False)
        predictions = list(self.model.predict(input_fn=predict_input_fn))

        return np.argmax(predictions[0][self.output_name.split('/')[0]])


    def input_predict_fn(self,features, target, df):
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype('category')
                mapp = {y: x for x, y in dict(enumerate(df[c].cat.categories)).items()}
                features[c] = float(mapp[features[c]])
            else:
                features[c] = float(features[c])
        del features[target]
        input_predict = np.fromiter(features.values(), dtype=float).reshape(1, -1)
        return input_predict

    def clear_checkpoint(self):
       shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def run(self):
       tf.estimator.train_and_evaluate(self.model, self.train_spec, self.eval_spec)

    def _create_model(self):
       self.keras_model = load_model('models/' +self.params['custom_model_path'])
       self.model = tf.keras.estimator.model_to_estimator(keras_model_path='models/' +self.params['custom_model_path'], config=self.runConfig) #TODO
       self.input_name = self.keras_model.inputs[0].name.split(':')[0]
       self.output_name = self.keras_model.outputs[0].name.split(':')[0]
       train_dataset, train_labels = self.train_csv_reader.make_numpy_array(self.train_csv_reader.label_name)
       val_dataset, val_labels = self.train_csv_reader.make_numpy_array(self.train_csv_reader.label_name)
       self._validation_input_fn = tf.estimator.inputs.numpy_input_fn(x={self.input_name: val_dataset}, y=val_labels, num_epochs=1, shuffle=True)
       self._train_input_fn = tf.estimator.inputs.numpy_input_fn(x={self.input_name: train_dataset}, y=train_labels, shuffle=True)

    def _create_specs(self):
       max_steps = self.params[MAX_STEPS]
       self.train_spec = tf.estimator.TrainSpec(
           input_fn=self._train_input_fn,
           max_steps=max_steps)

       self.eval_spec = tf.estimator.EvalSpec(
           input_fn=self._validation_input_fn,
           steps=None,  # How many batches of test data
           start_delay_secs=0, throttle_secs=1)


    def _create_run_config(self):
       save_checkpoints_steps = self.params[SAVE_CHECKPOINTS_STEPS]
       save_summary_steps = self.params[SAVE_SUMMARY_STEPS]
       keep_checkpoint_max = self.params[KEEP_CHECKPOINT_MAX]
       self.runConfig = tf.estimator.RunConfig(model_dir=self.checkpoint_dir,
                                               save_checkpoints_steps=save_checkpoints_steps,
                                               save_summary_steps=save_summary_steps,
                                               keep_checkpoint_max=keep_checkpoint_max)


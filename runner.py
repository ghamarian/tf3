from classifier import Classifier, KerasClassifier
from config import config_reader
from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader
import tensorflow as tf


class Runner:
    def __init__(self, config, feature_columns, label_name, label_unique_values, default_values, dtypes):
        self.config = config
        self.train_csv_reader = TrainCSVReader(self.config, default_values, dtypes, label_name)
        self.validation_csv_reader = ValidationCSVReader(self.config, default_values, dtypes, label_name)
        self.feature_columns = feature_columns
        self.label_unique_values = label_unique_values
        self.create_classifier()

    def create_classifier(self):
        params = {
                  'max_steps': 5000
                  }
        config_params = self.config.all()
        config_params.update(params)
        classifier = Classifier
        if config_params['custom_model_path'] !='None':
            classifier = KerasClassifier

        self.classifier = classifier(config_params, self.train_csv_reader, self.validation_csv_reader,
                                     self.feature_columns, self.label_unique_values)

    def run(self):
        try:
            self.classifier.run()
        # except tf.errors.NotFoundError:
        except:
            self.classifier.clear_checkpoint()
            self.classifier.run()

    def predict(self, features, target, df):
        try:
            result = self.classifier.predict(features, target, df)
        except:
            result = None
        return result

from classifier import Classifier
from config import config_reader
from train_csv_reader import TrainCSVReader
from validation_csv_reader import ValidationCSVReader
import tensorflow as tf


# CONFIG_FILE = "config/default.ini"
# config = config_reader.read_config(CONFIG_FILE)

class Runner:
    def __init__(self, config, feature_columns):
        self.config = config
        self.train_csv_reader = TrainCSVReader(self.config)
        self.validation_csv_reader = ValidationCSVReader(self.config)
        self.feature_columns = feature_columns

    def create_classifier(self):
        params = {'batch_size': 32,
                  'max_steps': 5000,
                  'save_checkpoints_steps': 100,
                  'save_summary_steps': 100,
                  'keep_checkpoint_max': 5,
                  'num_epochs': 200,
                  'validation_batch_size': 300
                  }
        config_params = self.config.all()
        config_params.update(params)
        self.classifier = Classifier(config_params, self.train_csv_reader, self.validation_csv_reader,
                                     self.feature_columns)

    def run(self):
        self.classifier.clear_checkpoint()
        self.classifier.run()

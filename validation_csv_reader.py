from csv_reader import CSVReader
from utils import abs_path_of
from typing import Dict


class ValidationCSVReader(CSVReader):

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.filename = abs_path_of(self.get_from_config('PATHS', 'validation_file'))
        self.label_name = self.get_label_name()
        self.num_epochs = 1

    def set_params(self, params: Dict[str, object]) -> None:
        self.batch_size = params.get('batch_size', self.get_int_from_config('TRAINING', 'validation_batch_size'))

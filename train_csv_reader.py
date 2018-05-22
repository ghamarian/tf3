from csv_reader import CSVReader
from utils import abs_path_of
from typing import Dict

class TrainCSVReader(CSVReader):

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.filename = abs_path_of(self._get_from_config('PATHS', 'training_file'))
        self.label_name = self._get_label_name()

    def _set_params(self, params: Dict[str, object]) -> None:
        self.batch_size = params.get('batch_size', self._get_int_from_config('TRAINING', 'batch_size'))
        self.num_epochs = params.get('num_epochs', self._get_int_from_config('TRAINING', 'num_epochs'))

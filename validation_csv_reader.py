from csv_reader import CSVReader
from utils import abs_path_of
from typing import Dict


class ValidationCSVReader(CSVReader):

    def __init__(self, config: str):
        super().__init__(config)
        self.filename = self.config.validation_path()
        self.label_name = self._get_label_name()
        self.num_epochs = 1

    def _set_params(self, params: Dict[str, object]) -> None:
        self.batch_size = params.get('validation_batch_size', self.config.validation_batch_size())


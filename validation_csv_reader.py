from csv_reader import CSVReader
from typing import Dict


class ValidationCSVReader(CSVReader):

    def __init__(self, config: str, label_name: str = None):
        super().__init__(config)
        self.filename = self.config.validation_path()
        if label_name is None:
            self.label_name = self._get_label_name()
        else:
            self.label_name = label_name
        self.num_epochs = 1

    def _set_params(self, params: Dict[str, object]) -> None:
        self.batch_size = params['validation_batch_size']

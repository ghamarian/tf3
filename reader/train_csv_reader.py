from reader.csv_reader import CSVReader
from typing import Dict

class TrainCSVReader(CSVReader):

    def __init__(self, config: str, column_defaults, dtypes, label_name: None):
        super().__init__(config, column_defaults, dtypes)
        self.filename = self.config.training_path()
        if label_name is None:
            self.label_name = self._get_label_name()
        else:
            self.label_name = label_name

    def _set_params(self, params: Dict[str, object]) -> None:
        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']

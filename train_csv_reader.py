from csv_reader import CSVReader
from typing import Dict

class TrainCSVReader(CSVReader):

    # def __init__(self, config: str):
    def __init__(self, config):
        super(TrainCSVReader, self).__init__(config)
        self.filename = self.config.training_path()
        self.label_name = self._get_label_name()

    # def _set_params(self, params: Dict[str, object]) -> None:
    def _set_params(self, params): # Dict[str, object] -> None:
        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']

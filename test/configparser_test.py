import config_reader


def test_configreader():
    config = config_reader.read_config("../config/default.ini")
    assert config['TASK0']['type'] == 'classification'
    assert config.get_as_slice("TASK0","ground_truth_column")  == -1

    print(config['PROCESS']['experiment_ID'])

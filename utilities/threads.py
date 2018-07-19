from dfweb import logging
from runner import Runner
import os
import time
import psutil


def tensor_board_thread(config_file, port, config_reader):
    # TODO testing to multiuser
    config_path = config_reader.read_config(config_file).all()['checkpoint_dir']
    logging.debug('Starting tensor board')
    time.sleep(3)
    os.system("tensorboard --logdir=" + config_path + "  --port=" + port)
    logging.debug('Exiting tensor board')


def run_thread(all_params_config, features, target, labels, defaults, dtypes):
    runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
    runner.run()


def pause_threads(username, processes):
    p = processes[username] if username in processes.keys() else None
    if not isinstance(p, str) and p:
        pid = p.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print
            "child", child
            child.kill()
        parent.kill()
        processes[username] = ''
    return True

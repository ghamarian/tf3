# from dfweb import logging
from runner import Runner
import os
import time
import psutil
from multiprocessing import Manager, Process, Queue
from utils.sys_ops import find_free_port, change_checkpoints
from config import config_reader
import threading
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

class ThreadHandler:
    def __init__(self):
        self._processes = {}
        self._ports = {}
        self._return_queue = Queue()

    def add_port(self, username, config_file, port):
        self._ports[username + '_' + config_file] = port

    def get_port(self, username, config_file):
        return self._ports[username + '_' + config_file]

    def tensor_board_thread(self, config_file, port):
        config_path = config_reader.read_config(config_file).all()['checkpoint_dir']
        logging.debug('Starting tensor board')
        time.sleep(3)
        pro = "tensorboard --host=0.0.0.0 --logdir=" + config_path + " --port=" + port
        subprocess.call(pro, shell=True)
        logging.debug('Exiting tensor board')

    def run_tensor_board(self, username, config_file):
        if not username + '_' + config_file in self._ports.keys():
            port = find_free_port()
            self.add_port(username, config_file, port)
            name = 'tensorboard-' + str(port)
            tboard_thread = threading.Thread(name=name,
                                             target=lambda: self.tensor_board_thread(config_file, port))
            tboard_thread.setDaemon(True)
            tboard_thread.start()

    def run_thread(self, all_params_config, features, target, labels, defaults, dtypes):
        runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
        runner.run()

    def predict_thread(self, all_params_config, features, target, labels, defaults, dtypes, new_features, df):
        runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
        self._return_queue.put(runner.predict(new_features, target, df))

    def explain_thread(self, all_params_config, features, target, labels, defaults, dtypes, new_features, df,
                       feature_types, num_features, top_labels):
        runner = Runner(all_params_config, features, target, labels, defaults, dtypes)
        self._return_queue.put(runner.explain(new_features, target, df, feature_types, num_features, top_labels))

    def pause_threads(self, username):
        p = self._processes[username] if username in self._processes.keys() else None
        if not isinstance(p, str) and p:
            pid = p.pid
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            del self._processes[username]
        return True

    def run_estimator(self, all_params_config, features, target, labels, defaults, dtypes, username):
        r_thread = Process(
            target=lambda: self.run_thread(all_params_config, features, target, labels, defaults, dtypes), name='run')
        r_thread.daemon = True
        r_thread.start()
        self._processes[username] = r_thread

    def predict_estimator(self, all_params_config, features, target, labels, defaults, dtypes, new_features, df):
        r_thread = Process(target=lambda: self.predict_thread(all_params_config, features, target,
                                                              labels, defaults, dtypes, new_features,
                                                              df), name='predict')
        r_thread.daemon = True
        r_thread.start()
        final_pred = self._return_queue.get()
        r_thread.join()
        return final_pred

    def explain_estimator(self, all_params_config, features, target, labels, defaults, dtypes, new_features, df,
                          feature_types, num_features, top_labels):
        r_thread = Process(target=lambda: self.explain_thread(all_params_config, features, target,
                                                              labels, defaults, dtypes, new_features,
                                                              df, feature_types, num_features, top_labels),
                           name='explain')
        r_thread.daemon = True
        r_thread.start()
        exp = self._return_queue.get()
        r_thread.join()
        return exp

    def handle_request(self, option, all_params_config, features, target, labels, defaults, dtypes, username,
                       resume_from):
        if option == 'run':
            if resume_from != '':
                change_checkpoints(all_params_config, resume_from)
            self.run_estimator(all_params_config, features, target, labels, defaults, dtypes, username)
        elif option == 'pause':
            self.pause_threads(username)
        else:
            raise ValueError("Invalid option")



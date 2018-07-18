from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_socketio import SocketIO, send
import time
import os


class Log:
    def __init__(self, logfile):
        self.app_socket = Flask(__name__)
        self.socketio = SocketIO(self.app_socket)
        WTF_CSRF_SECRET_KEY_2 = os.urandom(42)
        self.app_socket.secret_key = WTF_CSRF_SECRET_KEY_2
        self.logfile = logfile

    @socketio.on('disconnect')
    def test_disconnect(self):
        socketio.stop()
        connected = False
        print('no connected')


    @socketio.on_error() # Handles the default namespace
    def error_handler(e):
        print('SOCKET ERROR')
        pass


    @socketio.on('message')
    def handleMessage(self):
        while not os.path.isfile(self.logfile):
            time.sleep(0.1)
        import tailer
        for line in tailer.follow(open(self.logfile)):
            print(line)
            # if not connected:
            #     break
            if line is not None:
                time.sleep(0.1)
                send(line+'\n', broadcast=False)
            else:
                time.sleep(0.2)



    @socketio.on('my event')
    def test_message(self, message):
        print(message['data'])


    def run_app(self):
        self.socketio.run(self.app_socket, debug=True, port=61629)

#
#
# if __name__ == '__main__':
#     socketio.in
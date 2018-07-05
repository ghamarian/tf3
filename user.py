from flask_login import UserMixin

#TODO change to a database to save user data
class User(UserMixin):
    def __init__(self, username, password, email):
        self.id = username
        self.username = username
        self.password = password
        self.email = email
        self.configs = {}
        self.sessions = []

    def get_id(self):
        return self.id

    def add_config(self,config_name, config):
        self.configs[config_name] = config

    def get_config(self, config_name):
        return self.configs[config_name]

    def add_session(self,new_session):
        self.sessions.append(new_session)

    def is_session(self, session):
        return self.id if session in self.sessions else None


# from db import db
#
#
# class User(UserMixin, db.Model):
#     __tablename__ = 'users'
#
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(16), unique=True)
#     email = db.Column(db.String(50), unique=False)
#     password = db.Column(db.String(80))


import os
from user import User
from database.db import db
from werkzeug.security import check_password_hash


def get_db_user(username):
    return db.session.query(User.id).filter_by(username=username).scalar()


def get_user_by_username(username):
    return User.query.filter_by(username=username).first()


def create_user_path(username):
    if not os.path.exists(os.path.join('user_data/', username)):
        os.mkdir(os.path.join('user_data/', username))


def checklogin(username, password, remember, login_user, session, sess):
    new_user = get_db_user(username)
    if new_user is None:
        return False
    user = get_user_by_username(username)
    if check_password_hash(user.password, password):
        login_user(user, remember=remember)
        session['user'] = user.username
        sess.add_user(user.username)
        create_user_path(user.username)
        return True
    return False

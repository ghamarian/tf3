import os
from user import User
from db import db
from werkzeug.security import check_password_hash


def checklogin(_username, _password, _remember, login_user, session, sess):
    new_user = db.session.query(User.id).filter_by(username=_username).scalar()
    if new_user is None:
        return False
    user = User.query.filter_by(username=_username).first()
    if check_password_hash(user.password, _password):
        login_user(user, remember=_remember)
        session['user'] = user.username
        sess.add_user(user.username)
        if not os.path.exists(os.path.join('user_data/', user.username)):
            os.mkdir(os.path.join('user_data/', user.username))
        return True
    return False

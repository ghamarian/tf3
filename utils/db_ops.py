import os
from user import User
from database.db import db
from werkzeug.security import check_password_hash


def checklogin(username, password, remember, login_user, session, sess):
    new_user = db.session.query(User.id).filter_by(username=username).scalar()
    if new_user is None:
        return False
    user = User.query.filter_by(username=username).first()
    if check_password_hash(user.password, password):
        login_user(user, remember=remember)
        session['user'] = user.username
        sess.add_user(user.username)
        if not os.path.exists(os.path.join('user_data/', user.username)):
            os.mkdir(os.path.join('user_data/', user.username))
        return True
    return False

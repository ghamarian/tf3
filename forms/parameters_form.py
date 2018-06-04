from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField
from flask_uploads import UploadSet, DATA
from wtforms.validators import InputRequired

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params


class ParametersForm(FlaskForm):

    type = StringField("type", validators=[InputRequired()])
    submit = SubmitField("Submit")

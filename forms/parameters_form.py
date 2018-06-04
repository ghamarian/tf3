from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField, FormField, FileField
from flask_uploads import UploadSet, DATA
from wtforms.validators import InputRequired

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params


class ParametersForm(FlaskForm):
    type = StringField("type", validators=[InputRequired()])

class CheckpointsForm(FlaskForm):
    type = FileField("checkpoints", validators=[InputRequired()])
    somethingStupidd = StringField("Something", validators=[InputRequired()])


class GeneralForm(FlaskForm):
    parameters = FormField(ParametersForm)
    checkpoints = FormField(CheckpointsForm)

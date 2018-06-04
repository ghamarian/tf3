from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField, FormField, FileField, IntegerField, FieldList
from flask_uploads import UploadSet, DATA
from wtforms.validators import InputRequired

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params


class ParametersForm(FlaskForm):
    type = StringField("type", validators=[InputRequired()])


class CheckpointsForm(FlaskForm):
    checkpoint_dir = StringField("Checkpoints path", validators=[InputRequired()], default="checkpoints")
    log_dir = StringField("Log directory", validators=[InputRequired])


class ProcessForm(FlaskForm):
    keep_checkpoint_max = IntegerField("Maximum # of checkpoints", validators=[InputRequired()], default=50)
    save_checkpoints_steps = IntegerField("Save checkpoints after", validators=[InputRequired()], default=200)
    initialize_with_checkpoint = FileField("Initialize with checkpoints")
    save_summary_steps = IntegerField("Save summary after", validators=[InputRequired()], default=10)
    throttle = IntegerField("Evaluate after (s)", validators=[InputRequired()], default=1)


class LayerForm(FlaskForm):
    layer = IntegerField("Layer")


class NetworkForm(FlaskForm):
    num_layers = IntegerField("Number of layers", validators=[IntegerField()], default=1)
    # hidden_layers = FieldList(FormField(LayerForm), min_entries=3, default=(10, 5, 1))

class NetworkForm2(FlaskForm):
    num_layers = IntegerField("Number of layers", validators=[IntegerField()], default=1)
    # hidden_layers = FieldList(FormField(LayerForm), min_entries=3, default=(10, 5, 1))

class GeneralForm(FlaskForm):
    parameters = FormField(ParametersForm)
    checkpoints = FormField(CheckpointsForm)
    experiment = FormField(ProcessForm)
    network = FormField(NetworkForm)
    amir = FormField(NetworkForm2)


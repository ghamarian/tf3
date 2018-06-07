from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField, FormField, FileField, IntegerField, FieldList, SelectField, FloatField
from flask_uploads import UploadSet, DATA
from wtforms.validators import InputRequired, ValidationError, StopValidation, AnyOf, Regexp, NumberRange

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params


class PathsForm(FlaskForm):
    checkpoint_dir = StringField("Checkpoints path", validators=[InputRequired()], default="checkpoints")
    log_dir = StringField("Log directory", validators=[InputRequired()], default='checkpoints')


class ExperimentForm(FlaskForm):
    keep_checkpoint_max = IntegerField("Maximum # of checkpoints", validators=[InputRequired()], default=50)
    save_checkpoints_steps = IntegerField("Save checkpoints after", validators=[InputRequired()], default=200)
    initialize_with_checkpoint = FileField("Initialize with checkpoints")
    save_summary_steps = IntegerField("Save summary after", validators=[InputRequired()], default=10)
    throttle = IntegerField("Evaluate after (s)", validators=[InputRequired()], default=1)


class LayerForm(FlaskForm):
    layer = IntegerField("Layer")


def sanity_check_number_of_layers(form, field):
    if form.num_layers.data != len(form.hidden_layers.data.split(',')):
        field.errors[:] = []
        raise StopValidation('The numbers do not match')


class NetworkClassifierForm(FlaskForm):
    num_layers = IntegerField("Number of layers", validators=[InputRequired()], default=3)
    hidden_layers = StringField("Hidden units in csv",
                                validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$'), sanity_check_number_of_layers],
                                default="10,5,1")

    model_name = SelectField('Model type',
                             choices=[('DNNClassifier', 'DNN Classifier'), ('LinearClassifier', 'Linear Classifier')],
                             default='LinearClassifier')


class NetworkRegressorForm(FlaskForm):
    # num_layers = IntegerField("Number of layers", validators=[InputRequired()], default=3)
    # hidden_layers = StringField("Hidden units in csv", validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$'), sanity_check_number_of_layers], default="10,5,1")
    hidden_layers = StringField("Hidden units in csv", validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$')],
                                default="10,5,1")

    model_name = SelectField('Model type',
                             choices=[('DNNRegressor', 'DNN Regressor'), ('LinearRegressor', 'Linear Regressor')])


class TrainForm(FlaskForm):
    num_epochs = IntegerField("Number of epochs", validators=[InputRequired()], default=100)
    batch_size = IntegerField("Batch size", validators=[InputRequired()], default=32)
    optimizer = SelectField("Optimizer",
                            choices=[('Adagrad', 'Adagrad'), ('Adam', 'Adam'), ('Ftrl', 'Ftrl'), ('RMSProp', 'RMSProp'),
                                     ('SGD', 'SGD')])

    learning_rate = FloatField("Learning rate", validators=[InputRequired()], default=0.01)
    l1_regularization = FloatField("L1 regularization factor", validators=[InputRequired()], default=0.002)
    l2_regularization = FloatField("L2 regularization factor", validators=[InputRequired()], default=0.002)
    dropout_probability = FloatField("Dropout probability", validators=[InputRequired(), NumberRange(min=0.0, max=1.0)],
                                     default=0.0)


class GeneralRegressorForm(FlaskForm):
    checkpoints = FormField(PathsForm)
    experiment = FormField(ExperimentForm)
    network = FormField(NetworkClassifierForm)
    train = FormField(TrainForm)
    submit = SubmitField("Submit")


class GeneralClassifierForm(FlaskForm):
    paths = FormField(PathsForm)
    experiment = FormField(ExperimentForm)
    network = FormField(NetworkRegressorForm)
    train = FormField(TrainForm)
    submit = SubmitField("Submit")

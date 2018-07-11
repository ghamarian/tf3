from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField, FormField, FileField, IntegerField, FieldList, SelectField, FloatField
from flask_uploads import UploadSet, DATA
from wtforms.validators import InputRequired, ValidationError, StopValidation, AnyOf, Regexp, NumberRange

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params
import os


def sanity_check_number_of_layers(form, field):
    if form.num_layers.data != len(form.hidden_layers.data.split(',')):
        field.errors[:] = []
        raise StopValidation('The numbers do not match')

class PathsForm(FlaskForm):
    fcheckpoint_dir = StringField("Checkpoints path", validators=[InputRequired()], default="checkpoints")
    flog_dir = StringField("Log directory", validators=[InputRequired()], default='log')


class ExperimentForm(FlaskForm):
    keep_checkpoint_max = IntegerField("Maximum # of checkpoints", validators=[InputRequired()], default=50,
                                       description="MAXIMUN # OF CHECKPOINTS : The maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, all checkpoint files are kept.")
    save_checkpoints_steps = IntegerField("Save checkpoints after", validators=[InputRequired()], default=20,
                                          description="SAVE CHECKPOINTS STEPS : The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.")
    initialize_with_checkpoint = FileField("Initialize with checkpoints",
                                           description="INITIALIZE WITH CHECKPOINT : Directory where checkpoints model are saved.")
    save_summary_steps = IntegerField("Save summary after", validators=[InputRequired()], default=10,
                                      description="SAVE SUMMARY AFTER: Save summaries every this many steps.")
    throttle = IntegerField("Evaluate after (s)", validators=[InputRequired()], default=1,
                            description="EVALUATE AFTER (S):  Do not re-evaluate unless the last evaluation was started at least this many seconds ago. Of course, evaluation does not occur if no new checkpoints are available, hence, this is the minimum.")
    validation_batch_size = IntegerField("Validation batch size", validators=[InputRequired()], default=32,
                                         description="VALIDATION BATCH SIZE: ")


class LayerForm(FlaskForm):
    layer = IntegerField("Layer")


class NetworkClassifierForm(FlaskForm):
    # num_layers = IntegerField("Number of layers", validators=[InputRequired()], default=3)
    hidden_layers = StringField("Hidden units in csv",
                                validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$')],
                                default="10,5,1",
                                description="HIDDEN UNITS IN CSV: ")

    model_name = SelectField('Model type',
                             choices=[('DNNClassifier', 'DNN Classifier'),
                                      ('LinearClassifier', 'Linear Classifier'),
                                      ('DNNLinearCombinedClassifier', 'DNN Linear Combined Classifier')],
                             default='DNN Classifier',description="MODEL TYPE: ")


class NetworkRegressorForm(FlaskForm):
    # num_layers = IntegerField("Number of layers", validators=[InputRequired()], default=3)
    # hidden_layers = StringField("Hidden units in csv", validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$'), sanity_check_number_of_layers], default="10,5,1")
    hidden_layers = StringField("Hidden units in csv", validators=[InputRequired(), Regexp(r'\d+(?:,\d+)*$')],
                                default="10,5,5",
                                description="HIDDEN UNITS: ")
    model_name = SelectField('Model type',
                             choices=[('DNNRegressor', 'DNN Regressor'),
                                      ('LinearRegressor', 'Linear Regressor'),
                                      ('DNNLinearCombinedRegressor', 'DNN Linear Combined Regressor')],
                             description="MODEL TYPE: ")

class CustomModelForm(FlaskForm):
    models = os.listdir('models')
    choices = [(x,x) for x in models]
    choices.append(('None','None'))
    custom_model_path = SelectField('Custom model',
                             choices= choices, default='None',
                                    description="CUSTOM MODEL: ")

class TrainForm(FlaskForm):
    num_epochs = IntegerField("Number of epochs", validators=[InputRequired()], default=100,
                              description="NUMBER OF EPOCHS: ")
    batch_size = IntegerField("Batch size", validators=[InputRequired()], default=32,
                              description="BATCH SIZE: ")
    optimizer = SelectField("Optimizer",
                            choices=[('Adagrad', 'Adagrad'), ('Adam', 'Adam'), ('Ftrl', 'Ftrl'), ('RMSProp', 'RMSProp'),
                                     ('SGD', 'SGD')], default='Adam',
                                    description="OPTIMIZER: provides methods to compute gradients for a loss and apply gradients to variables")

    learning_rate = FloatField("Learning rate", validators=[InputRequired()], default=0.01)
    l1_regularization = FloatField("L1 regularization factor", validators=[InputRequired()], default=0,
                                   description="L1 REGULARIZATION: ")
    l2_regularization = FloatField("L2 regularization factor", validators=[InputRequired()], default=0,
                                   description="L2 REGULARIZATION: ")
    dropout = FloatField("Dropout probability", validators=[InputRequired(), NumberRange(min=0.0, max=1.0)],
                                     default=0.0, description="DROPOUT: ")

    activation_fn = SelectField("Activation function",
                                choices=[('relu', 'ReLU'), ('tanh', 'Hyperbolic Tangent'),
                                         ('sigmoid', 'Sigmoid')], default='relu',
                                description="ACTIVATION FUNCTION: ")


class GeneralRegressorForm(FlaskForm):
    # paths = FormField(PathsForm)
    experiment = FormField(ExperimentForm)
    network = FormField(NetworkRegressorForm)
    custom_model = FormField(CustomModelForm)
    training = FormField(TrainForm)
    submit = SubmitField("Submit")


class GeneralClassifierForm(FlaskForm):
    # paths = FormField(PathsForm)
    experiment = FormField(ExperimentForm)
    network = FormField(NetworkClassifierForm)
    custom_model = FormField(CustomModelForm)
    training = FormField(TrainForm)
    submit = SubmitField("Submit")

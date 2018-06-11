from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed, FileField
from wtforms import SubmitField, SelectField, FormField, SelectMultipleField, BooleanField
from flask_uploads import UploadSet, DATA

from wtforms import StringField
from wtforms.validators import InputRequired
from wtforms.widgets import HTMLString, html_params
import os

files = os.listdir('datasets')


# To avoid the limitation of FormField of wtforms
class FileInputWithAccept:
    def __call__(self, field, **kwargs):
        kwargs.setdefault('id', field.id)
        return HTMLString(
            '<input %s>' % html_params(label=field.label, name=field.name, type='file', accept='text/csv', **kwargs))


class FileFieldWithAccept(StringField):
    widget = FileInputWithAccept()


dataset = UploadSet(extensions=DATA)


class NewFileForm(FlaskForm):
    train_file = FileFieldWithAccept(label='Train dataset in CSV format',
                                     validators=[FileRequired(),
                                                 FileAllowed(['csv'], message="Please enter csv file.")])

    test_file = FileFieldWithAccept(label='Test dataset in CSV format',
                                    validators=[FileAllowed(['csv'], message="Please enter csv file.")])

    # submit = SubmitField("Submit")


class ExisitingDatasetForm(FlaskForm):
    train_file = SelectField(u'Field name', choices=list(zip(files, files)), validators=[InputRequired()])
    test_file = SelectField(u'Field name', choices=list(zip(files, files)), validators=[InputRequired()])


# class ExistingTestDatasetForm(FlaskForm):
#     test_file = SelectField(u'Field name', choices=list(zip(files, files)), validators=[InputRequired()])


class UploadForm(FlaskForm):
    old_or_new = BooleanField('Existing dataset', validators=[InputRequired()])
    new_files = FormField(NewFileForm)
    exisiting_files = FormField(ExisitingDatasetForm)
    # exisiting_test_file = FormField(ExistingTestDatasetForm)
    submit = SubmitField("Submit")

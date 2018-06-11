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
                                     validators=[FileAllowed(['csv'], message="Please enter csv file.")])

    test_file = FileFieldWithAccept(label='Test dataset in CSV format',
                                    validators=[FileAllowed(['csv'], message="Please enter csv file.")])


class ExisitingDatasetForm(FlaskForm):
    train_file = SelectField(u'Train dataset', choices=list(zip(files, files)))
    test_file = SelectField(u'Test dataset', choices=list(zip(files, files)))


class UploadForm(FlaskForm):
    is_existing = BooleanField('Existing dataset')
    new_files = FormField(NewFileForm)
    exisiting_files = FormField(ExisitingDatasetForm)
    # exisiting_test_file = FormField(ExistingTestDatasetForm)
    submit = SubmitField("Submit")

from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed, FileField
from wtforms import SubmitField, SelectField, FormField, SelectMultipleField, BooleanField
from flask_uploads import UploadSet, DATA
from wtforms import StringField
from wtforms.widgets import HTMLString, html_params

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

    test_file = FileFieldWithAccept(label='Validation dataset in CSV format',
                                    validators=[FileAllowed(['csv'], message="Please enter csv file.")] )


class ExisitingDatasetForm(FlaskForm):
    train_file_exist = SelectField(u'Train dataset')


class UploadForm(FlaskForm):
    is_existing = BooleanField('Existing dataset')
    new_files = FormField(NewFileForm)
    exisiting_files = FormField(ExisitingDatasetForm)
    submit = SubmitField("Submit")


class UploadNewForm(FlaskForm):
    new_files = FormField(NewFileForm)
    submit = SubmitField("Submit")
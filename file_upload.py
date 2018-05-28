from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import SubmitField
from flask_uploads import UploadSet, DATA

from wtforms import StringField
from wtforms.widgets import HTMLString, html_params


# To avoid the limitation of FormField of wtforms
class FileInputWithAccept:
    def __call__(self, field, **kwargs):
        kwargs.setdefault('id', field.id)
        return HTMLString('<input %s>' % html_params(name=field.name, type='file', accept='text/csv', **kwargs))


class FileFieldWithAccept(StringField):
    widget = FileInputWithAccept()


dataset = UploadSet(extensions=DATA)


class FileForm(FlaskForm):
    file = FileFieldWithAccept(label='dataset in csv',
                               validators=[FileRequired(), FileAllowed(['csv'], message="Please enter csv file.")])
    submit = SubmitField("Submit")

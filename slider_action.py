from flask_wtf import FlaskForm
from wtforms import SubmitField, BooleanField
from wtforms.widgets import HTMLString, html_params, Input


# class FileInputWithAccept:
#     def __call__(self, field, **kwargs):
#         kwargs.setdefault('id', field.id)
#         return HTMLString(
#             '<input %s>' % html_params(label=field.label, name=field.name, type='file', accept='text/csv', **kwargs))

class DisabledSubmitInput(Input):
    """
    Renders a submit button.

    The field's label is used as the text of the submit button instead of the
    data on the field.
    """
    input_type = 'submit'

    def __call__(self, field, **kwargs):
        kwargs.setdefault('value', field.label.text)
        return super(DisabledSubmitInput, self).__call__(field, disabled=True, **kwargs)


class DisabledSubmit(BooleanField):
    widget = DisabledSubmitInput()

class SliderSubmit(FlaskForm):
    submit = DisabledSubmit("Next", disabled=True)

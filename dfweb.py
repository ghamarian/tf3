from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from flask_bootstrap import Bootstrap
import os

from werkzeug.utils import secure_filename

from file_upload import FileForm

WTF_CSRF_SECRET_KEY = 'a random string'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
Bootstrap(app)
app.secret_key = WTF_CSRF_SECRET_KEY


# @app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/')
def analysis():
    # return render_template('dataset_upload.html', name='Upload dataset')
    return redirect(url_for('upload'))
    # x = pd.DataFrame(np.random.randn(20, 5))
    # return render_template("analysis.html", name='amir', data=x)


@app.route('/slider')
def slider():
    return render_template("slider.html")


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % error)
            # flash(u"Error in the %s field - %s" % (getattr(form, field).label.text, error))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = FileForm()
    form.file()
    if form.validate_on_submit():
        f = form.file.data
        filename = secure_filename(f.filename)

        target = os.path.join(APP_ROOT, "datasets")
        if not os.path.isdir(target):
            os.mkdir(target)

        destination = os.path.join(target, filename)

        f.save(destination)
        return redirect(url_for('slider'))
    flash_errors(form)
    return render_template('upload_file_wtf.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)

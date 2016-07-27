from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
import os
from config import basedir

app = Flask(__name__)
app.config.from_object('config')
app.config['MAX_CONTENT_LENGTH'] = 5.0 * 1024 * 1024
db = SQLAlchemy(app)

from app import views, models, classify_image


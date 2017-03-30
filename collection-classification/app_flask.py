#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os

from flask import Flask
from flask_cors import CORS, cross_origin

from app import context
from app.core.json import CustomJSONEncoder
from app.rest import controllers

# keras CPU only  mode.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

LOGGER = logging.getLogger(__name__)

context.Initializer().initialize()

app = Flask(__name__, static_folder="")
CORS(app)

app.json_encoder = CustomJSONEncoder
app.config.update(context.get('app.flask', dict))

controllers.register(app)
context.put('flask.app', app)

app.run(host="0.0.0.0", threaded=True)

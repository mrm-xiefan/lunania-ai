#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os
import uuid
from io import BytesIO

from flask import Blueprint, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from app import context
from app.core.decorator.http import crossdomain, gzipped
from app.models.manager import ModelStore
from app.constants.color_palette import LabelEnum
from app.services.fcn import FcnService
from app.utils import rest_utils
from typing import Dict

LOGGER = logging.getLogger(__name__)

app = Blueprint("fcn", __name__, url_prefix="/fcn")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PREDICTION_RESULT_EXTENSION = '.jpg'


def _get_upload_base_folder():
    '''
    Get flask.config[UPLOAD_FOLDER]
    '''
    return context.get('flask.app', None).config['UPLOAD_FOLDER']


def _get_file_store_path(model_key: str, file_key: str, ext: str) -> str:
    model_folder = os.path.join(_get_upload_base_folder(), model_key)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    return os.path.join(model_folder, file_key + ext)


def _allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/<model_key>/prediction", methods=['POST', 'OPTIONS'])
@crossdomain(origin=['*'])
def predict_from_upload_file(model_key: str):
    model_store = context.get(
        context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and _allowed_file(file.filename):
        # create unique file_key
        file_key = str(uuid.uuid4())
        file_name, file_ext = os.path.splitext(secure_filename(file.filename))

        # save upload file
        file_path = _get_file_store_path(model_key, file_key, file_ext)
        file.save(file_path)

        # predict
        size = context.get('tag.fcn.size.eval', tuple)
        classes = context.get('tag.fcn.classes', int)
        fcn_service = FcnService(model_store)
        result, crf_result = fcn_service.predict(
            model_key, file_path, size, classes, print_summary=False)

        label = LabelEnum.of(classes)
        label.to_image(result).save(_get_file_store_path(model_key,
                                                         file_key + '-org',
                                                         PREDICTION_RESULT_EXTENSION))

        label.to_image(crf_result).save(_get_file_store_path(model_key,
                                                             file_key + '-crf',
                                                             PREDICTION_RESULT_EXTENSION))

        return jsonify({
            'predictUrl': '/'.join(['fcn', model_key, 'prediction', file_key]),
            'suffixList': ['source', 'org', 'crf']
        })


@app.route("/<model_key>/prediction/<file_key>", methods=['GET', 'OPTIONS'])
@crossdomain(origin=['*'])
def get_predict_file(model_key: str, file_key: str):
    # get prediction source

    if file_key.endswith('.source'):
        file_key_suffix = ''
    elif file_key.endswith('.org'):
        file_key_suffix = '-org'
    else:
        file_key_suffix = '-crf'
        file_key = file_key + '.crf'

    LOGGER.debug('suffix:%s', file_key_suffix)

    file_key = file_key[:file_key.rfind('.')]
    base_folder = os.path.join(_get_upload_base_folder(), model_key)
    return send_from_directory(base_folder,
                               file_key + file_key_suffix +
                               PREDICTION_RESULT_EXTENSION)

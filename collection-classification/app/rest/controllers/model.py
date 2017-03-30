#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
from io import BytesIO

from flask import Blueprint, jsonify, request, send_file

from app import context
from app.core.decorator.http import crossdomain, gzipped
from app.models.manager import ModelStore, LearningHistory
from app.utils import rest_utils
from typing import Dict

LOGGER = logging.getLogger(__name__)

app = Blueprint("model", __name__, url_prefix="/models")

# default model store.
_model_store = model_store = context.get(
    context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore


@app.route("", methods=['GET', 'OPTIONS'])
@crossdomain(origin=['*'])
@gzipped
def get_models():
    '''
    Get saved model keys.
    '''
    model_keys = _model_store.get_model_keys()
    return jsonify({
        'modelKeys': model_keys
    })


@app.route("/<model_key>/definition", methods=['GET', 'OPTIONS'])
@crossdomain(origin=['*'])
@gzipped
def get_model_definition(model_key: str):
    '''
    Get saved model definition.
    '''
    ret = _model_store.get_model_definition(model_key)
    if ret is None:
        # TODO return 404.
        pass
    LOGGER.debug(ret)
    return jsonify(ret)


@app.route("/<model_key>/histories", methods=['GET', 'OPTIONS'])
@crossdomain(origin=['*'])
@gzipped
def get_histories(model_key: str):
    LOGGER.debug('param modelkey:%s', model_key)
    # model_keys = request.args.get('modelKeys', default='', type=str)
    # model_keys = [] if not model_keys else model_keys.split(',')

    LOGGER.debug('url_root:%s', request.url_root)
    history_list = _model_store.get_histories([model_key])
    history = history_list[0]
    return jsonify(to_dto(history))


def to_learning_history(ob):
    model_key = ob['modelKey']
    ob['visualizedModelUrl'] = request.url_root + \
        'models/' + model_key + '/visualization'

    return ob


def to_dto(history: LearningHistory) -> Dict[str, any]:
    return to_learning_history(
        rest_utils.to_camel_case_dict_key(history.__dict__, {}))


@app.route("/<model_key>/visualization", methods=['GET', 'OPTIONS'])
@crossdomain(origin=['*'])
def get_model_image(model_key: str):

    image = _model_store.get_model_image(model_key)
    img_io = BytesIO()
    image.save(img_io, 'png')
    # required
    image.close()
    img_io.seek(0)
    file_name = model_key + ".png"
    LOGGER.debug('file_name:%s', file_name)
    return send_file(img_io,
                     attachment_filename=file_name,
                     as_attachment=False)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    LOGGER.debug('start add_header.')
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

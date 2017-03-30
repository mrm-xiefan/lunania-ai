#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import logging.config
import os
import sys
import ast

import yaml

from app.datasets.manager import DatasetStore
from app.models.manager import ModelStore
from app.core.log import StreamToLogger
from typing import TypeVar

LOGGER = logging.getLogger(__name__)

_APP_CONTEXT = {}
CONTEXT_KEY_APP_CONFIG = 'app.context.ApplicationConfig'
CONTEXT_KEY_MODEL_STORE = 'app.models.ModelStore'
CONTEXT_KEY_DATASET_STORE = 'app.datasets.DatasetStore'


class Initializer():

    def __init__(self):
        pass

    def initialize(self):

        # force update cwd.
        os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

        # initialize logging configuration.
        with open('app/resources/logging.yaml', 'r') as file:
            logging.config.dictConfig(yaml.load(file))

        LOGGER.info('context initialize start.')
        LOGGER.info('cwd:%s', os.getcwd())

        app_config = ApplicationConfig('app/resources/application.yaml')
        put(CONTEXT_KEY_APP_CONFIG, app_config)
        put(CONTEXT_KEY_MODEL_STORE, ModelStore(
            app_config.get('model.dir')))
        put(CONTEXT_KEY_DATASET_STORE, DatasetStore(
            app_config.get('train.dir')))

        for key in show():
            LOGGER.info('context load key:%s', key)

        # override print function
        sys.stdout = StreamToLogger(logging.getLogger('stdout'), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger('stderr'), logging.INFO)

        LOGGER.info('context initialize end.')


class ApplicationConfig():

    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.config = yaml.load(file)

    def get(self, keys):
        if isinstance(keys, str):
            keys = keys.split('.')

        last = keys[-1]
        ret = self.__get(self.config, keys)
        if last == 'dir':
            return os.path.expanduser(ret)
        if last == 'eval':
            return ast.literal_eval(ret)
        else:
            return ret

    def __get(self, config, keys):
        if len(keys) == 1:
            return config[keys[0]]
        else:
            return self.__get(config[keys[0]], keys[1:])


def put(key: str, value):
    _APP_CONTEXT[key] = value

T = TypeVar('T')


def get(key, ret: T) -> T:
    if key in _APP_CONTEXT:
        return _APP_CONTEXT[key]
    else:
        return _APP_CONTEXT[CONTEXT_KEY_APP_CONFIG].get(key)


def show():
    return _APP_CONTEXT.keys()

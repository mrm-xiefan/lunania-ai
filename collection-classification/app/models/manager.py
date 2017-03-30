#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import glob
import json
import logging
import os
from collections import ChainMap
from datetime import datetime as dt

import h5py
import yaml
from keras import models
from keras.callbacks import Callback, History
from keras.models import Model
from keras.utils import visualize_util
from PIL import Image

from app.core.callback import LearningTime, LearningTimeLogger
from typing import Dict, List

from . import ModelContainer, SavedModelContainer

LOGGER = logging.getLogger(__name__)


def support_leargninTimeLogger_default(o):
    if hasattr(o, '__dict__'):
        return o.__dict__
    else:
        return o

    raise TypeError(repr(o) + " is not JSON serializable")


class ModelStore():

    MODEL_FILE_NAME = 'model.yaml'
    VISUALIZED_MODEL_FILE_NAME = 'model.png'
    WEIGHTS_FILE_NAME = 'model.h5'
    ALL_IN_MODEL_FILE_NAME = 'model_all.h5'
    HISTORY_FILE_NAME = 'history.json'

    def __init__(self, base_folder_path: str):
        self._base_folder_path = base_folder_path

    def save(self, key: str, model: Model, history: History,
             fit_callbacks: [Callback]=None,
             prev_key: str='',
             is_override=False) -> str:

        if not is_override:
            key = key + dt.now().strftime('-%Y%m%d%H%M%S')

        folder = self._get_store_path(key)
        try:
            os.makedirs(folder)
        except FileExistsError:
            LOGGER.info('Exist Folder:%s. file overide.', key)

        # save model
        with open(os.path.join(folder, self.MODEL_FILE_NAME), mode='w', encoding='utf-8') as model_file:
            model_file.write(model.to_yaml())

        # save model visual
        visualize_util.plot(model, to_file=os.path.join(
            folder, self.VISUALIZED_MODEL_FILE_NAME), show_shapes=True)

        # save model weights
        model.save_weights(os.path.join(folder, self.WEIGHTS_FILE_NAME))

        # save all in package
        model.save(os.path.join(folder, self.ALL_IN_MODEL_FILE_NAME))

        # save history
        time_loggers = list(filter(
            lambda x: isinstance(x, LearningTimeLogger),
            fit_callbacks))
        learning_time_log = {} if len(time_loggers) == 0 else time_loggers[0]
        with open(os.path.join(folder, self.HISTORY_FILE_NAME),
                  mode='w', encoding='utf-8') as history_file:
            log = learning_time_log.__dict__
            del log['model']
            if 'lr' in history.history:
                del history.history['lr']
            json.dump(dict(**ChainMap(history.history,
                                      {'learning_time_log': log,
                                       'prev_model_key': prev_key})),
                      history_file, indent=4, default=support_leargninTimeLogger_default)

        return key

    def load(self, key: str, use_all_in_one=True) -> ModelContainer:

        exists_all_in_one = os.path.exists(os.path.join(self._get_store_path(key),
                                                        self.ALL_IN_MODEL_FILE_NAME))
        if exists_all_in_one and use_all_in_one:
            LOGGER.debug('use all in one.')
            model = models.load_model(os.path.join(self._get_store_path(key),
                                                   self.ALL_IN_MODEL_FILE_NAME))
        else:
            with open(os.path.join(self._get_store_path(key),
                                   self.MODEL_FILE_NAME), 'r') as yaml_string:
                model = models.model_from_yaml(yaml_string)  # type: Model
            model.load_weights(os.path.join(self._get_store_path(key),
                                            self.WEIGHTS_FILE_NAME), by_name=True)

        return SavedModelContainer(key, model, exists_all_in_one and use_all_in_one)

    def get_weights(self, key: str):
        f = h5py.File(os.path.join(
            self._get_store_path(key), self.WEIGHTS_FILE_NAME))
        return f

    def load_weights(self, model, key: str):
        model.load_weights(os.path.join(
            self._get_store_path(key), self.WEIGHTS_FILE_NAME), by_name=True)
        return model

    def get_model_keys(self) -> List[str]:
        '''
        Get saved model key.
        '''
        dir_paths = glob.glob(self._get_store_path('*'))
        return sorted(list(map(os.path.basename, dir_paths)), key=str)

    def _get_create_date_from_model_key(self, model_key: str) -> dt:
        path = self._get_store_path(model_key)
        return dt.fromtimestamp(os.stat(path).st_mtime)

    def get_histories(self, keys: List[str]=None) -> List['LearningHistory']:
        '''
        学習時のacc,loss等の履歴情報を取得する。
        '''

        if keys is None or len(keys) == 0:
            LOGGER.debug('load glob')
            dir_paths = glob.glob(self._get_store_path('*'))
        else:
            LOGGER.debug('load modelKey:%s', keys)
            dir_paths = [self._get_store_path(key) for key in keys]
            LOGGER.debug(dir_paths)

        ret = []  # type: List['LearningHistory']
        for dir_path in dir_paths:
            if os.path.isdir(dir_path):
                key = os.path.basename(dir_path)
                if os.path.exists(os.path.join(dir_path, self.HISTORY_FILE_NAME)):
                    with open(os.path.join(dir_path, self.HISTORY_FILE_NAME), mode='r', encoding='utf-8') as history_file:
                        LOGGER.debug('load file:%s', key)
                        history = json.load(history_file)
                        ret.append(LearningHistory(key, history))

        return ret

    def get_model_image(self, key: str) -> Image.Image:
        return Image.open(os.path.join(
            self._get_store_path(key), self.VISUALIZED_MODEL_FILE_NAME))

    def get_model_definition(self, key: str) -> dict:
        path = os.path.join(self._get_store_path(key), self.MODEL_FILE_NAME)
        if os.path.exists(path):
            with open(path, mode='r') as def_file:
                ret = yaml.load(def_file)
        else:
            ret = None

        return ret

    def _get_store_path(self, key: str):
        return os.path.join(self._base_folder_path, key)


class LearningHistory():

    HISTORY_DEFAULT_KEYS = ['acc', 'loss', 'val_acc', 'val_loss']

    def __init__(self, model_key: str, history: Dict[str, List[float]]):
        self.model_key = model_key
        self.history = history

        # set default value
        for key in self.HISTORY_DEFAULT_KEYS:
            if key not in self.history:
                self.history[key] = []

        self.epoch = len(self.history[self.HISTORY_DEFAULT_KEYS[0]])

#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import numpy as np

from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img

from app.core.callback import LearningTimeLogger
from app.datasets.manager import DatasetStore
from app.models.pattern import PatternModelContainer
from app.models.manager import ModelContainer, ModelStore
from app.utils import layer_utils
from typing import Tuple

LOGGER = logging.getLogger(__name__)


class PatternClassificationService():

    def __init__(self, model_store: ModelStore, dataset_store: DatasetStore):
        self._model_store = model_store  # type: ModelStore
        self._dataset_store = dataset_store  # type: DatasetStore

    def learn(self, model_container: PatternModelContainer,
              save_model_key: str, dataset_key: str,
              validation_dataset_key: str,
              input_shape: Tuple[int],
              nb_epoch: int,
              is_data_aug=False,
              aug_rate=1) -> str:

        LOGGER.debug('start. %s', model_container.get_model_key())
        # model get and compile
        model = model_container.get_model(input_shape)

        # set default weight and this layer set training false.
        # layer_utils.set_vgg16_weights(model)

        model_container.compile(model)

        return self._learn(model, save_model_key, dataset_key,
                           validation_dataset_key,
                           (300, 200, 3), nb_epoch,
                           is_data_aug=is_data_aug,
                           aug_rate=aug_rate)

    def _learn(self, model: Model,
               save_model_key: str, dataset_key: str,
               validation_dataset_key: str,
               input_shape: Tuple[int],
               nb_epoch: int,
               is_data_aug=False,
               aug_rate=1) -> str:

        # get learning data
        if is_data_aug:
            dataset = self._dataset_store.load(dataset_key,
                                               input_shape[:2],
                                               generator_option={
                                                   'rescale': 1. / 255
                                               },
                                               flow_option={
                                                   'batch_size': 32,
                                               })
        else:
            aug_rate = 1
            dataset = self._dataset_store.load(dataset_key, input_shape)

        # get validation data.
        validation_dataset = self._dataset_store.load(validation_dataset_key,
                                                      input_shape[:2],
                                                      generator_option={
                                                          'rescale': 1. / 255
                                                      })

        LOGGER.debug('epoch:%s, sample_count:%s',
                     nb_epoch, dataset.nb_sample)

        # save dataset indices.
        LOGGER.debug(dataset.class_indices)

        model.summary()
        # set default callbacks
        lt_cb = LearningTimeLogger(is_log_batch=False)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                       factor=np.sqrt(0.1), cooldown=0,
                                       patience=5, min_lr=0.5e-6)
        tensor_cb = TensorBoard()
        history = model.fit_generator(dataset, dataset.nb_sample,
                                      nb_epoch, verbose=1, callbacks=[lt_cb, lr_reducer, tensor_cb],
                                      validation_data=validation_dataset,
                                      nb_val_samples=validation_dataset.nb_sample)

        saved_model_key = self._model_store.save(
            save_model_key, model, history, fit_callbacks=[lt_cb])
        return saved_model_key

    def continue_learning(self, model_key: str,
                          dataset_key: str,
                          validation_dataset_key: str,
                          input_shape: Tuple[int],
                          nb_epoch: int,
                          is_data_aug=False,
                          aug_rate=1) -> str:
        LOGGER.debug('start. %s', model_key)

        model_container = self._model_store.load(
            model_key, use_all_in_one=True)
        model = model_container.get_model()

        return self._learn(model, model_key, dataset_key,
                           validation_dataset_key,
                           input_shape, nb_epoch,
                           is_data_aug=is_data_aug,
                           aug_rate=aug_rate)

    def predict(self, model_key: str, generator):
        model_container = self._model_store.load(
            model_key, use_all_in_one=False)

        model = model_container.get_model()

        result = model.predict_generator(generator, generator.nb_sample)
        return result

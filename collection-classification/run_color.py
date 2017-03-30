#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import shutil

from keras.backend import tensorflow_backend as tf
# keras import
from keras.preprocessing.image import load_img
from PIL import Image

# app initialize
from app import context
from app.datasets.manager import DatasetStore
from app.models.color import (ColorModelContainer,
                              ColorModelContainerAtResNet,
                              ColorModelContainerOn1Layer,
                              ColorModelContainerOn2Layer,
                              ColorModelContainerOn3Layer)
from app.models.manager import ModelStore
from app.prediction.processor import ClassificationProcessor
from app.services.color import ColorClassificationService
from app.core.log import logging_time
from app.utils import image_utils
from typing import List

LOGGER = logging.getLogger(__name__)


def continue_learning(epochs: List[int],
                      parent_model_key: str,
                      color_classification_service: ColorClassificationService):

    model_key = parent_model_key
    for epoch in epochs:
        LOGGER.debug('learn start:%s', model_key)
        model_key = color_classification_service.continue_learning(
            model_key,
            'color-20170208-10class-increase',
            'color-20170208-10class-validation',
            (300, 200, 3), epoch,
            is_data_aug=True, aug_rate=3)
        tf.clear_session()


def increase():

    org_dir = '/var/ria-dataset/app/train/color-20170214-11class'
    save_dir = '/var/ria-dataset/app/train/color-20170214-11class-increase'

    image_utils.increase_img_data(org_dir, save_dir, 1000,
                                  generator_option={
                                      'vertical_flip': True,
                                      'horizontal_flip': True,
                                      'width_shift_range': 0.3,
                                      'height_shift_range': 0.3,
                                      'zoom_range': 0.2
                                  },
                                  flow_option={
                                      'target_size': (300, 200)
                                  })


@logging_time
def predict_by_dir(dataset_store: DatasetStore, model_store: ModelStore,
                   model_key: str):

    target_dir = '/var/ria-dataset/archive/data/spring-2015-ready-to-wear/cropped_label/collection_label5/org/1'
    dist_dir = '/var/ria-dataset/archive/data/temp/2015'

    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)

    model_container = model_store.load(model_key, use_all_in_one=False)
    model = model_container.get_model()

    label = ['beige', 'black', 'blue', 'brown', 'gray', 'green',
             'navy', 'orange', 'pink', 'red', 'white']
    processor = ClassificationProcessor(label, threshold=0.65)

    counter = 0
    for file_path in glob.glob(os.path.join(target_dir, '**')):
        if os.path.isfile(file_path):
            LOGGER.debug(file_path)
            img = Image.open(file_path).resize((200, 300))  # type: Image.Image
            img = image_utils.to_array_and_reshape(img)
            result = model.predict_on_batch(img)

            ret = processor.process(result)
            dist_sub_dir = os.path.join(dist_dir, ret[0])
            if not os.path.exists(dist_sub_dir):
                os.mkdir(dist_sub_dir)

            shutil.copy(file_path, dist_sub_dir)
            counter += 1

    LOGGER.info('predict count:%s', str(counter))


def predict(dataset_store: DatasetStore, model_store: ModelStore, model_key: str):

    generator = dataset_store.load('color-20170214-11class-validation',
                                   (300, 200),
                                   generator_option={
                                       'rescale': 1. / 255
                                   }, flow_option={
                                       'shuffle': False
                                   })

    color_classification_service = ColorClassificationService(model_store,
                                                              dataset_store)

    label = ['beige', 'black', 'blue', 'brown', 'gray', 'green',
             'navy', 'orange', 'pink', 'red', 'white']

    processor = ClassificationProcessor(label)
    result = color_classification_service.predict(model_key,  generator)
    result = processor.process(result)

    with open('./color-result.txt', 'w') as f:
        f.writelines(list(map(lambda x: x + '\r\n', result)))


if __name__ == '__main__':
    context.Initializer().initialize()

    model_store = context.get(
        context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore
    dataset_store = context.get(
        context.CONTEXT_KEY_DATASET_STORE, DatasetStore)  # type: DatasetStore

    color_classification_service = ColorClassificationService(model_store,
                                                              dataset_store)

    # model_container = ColorModelConatainerAtResNet(11)
    # new_model_key = color_classification_service.learn(model_container,
    #                                                    model_container.get_model_key() + '-color-20170214-11class-increase',
    #                                                    'color-20170214-11class-increase',
    #                                                    'color-20170206-11class-validation',
    #                                                    (3, 300, 200), 30,
    # is_data_aug=True, aug_rate=3)

    # tf.clear_session()

    # new_model_key = 'ColorModelConatainerAtResNet-color-20170208-10class-increase-20170213235656'
    # continue_learning([10, 10], new_model_key,
    #                   color_classification_service)
    # increase()
    # predict(dataset_store, model_store,
    #         'ColorModelConatainerAtResNet-color-20170214-11class-increase-20170214091529')
    predict_by_dir(dataset_store, model_store,
                   'ColorModelConatainerAtResNet-color-20170214-11class-increase-20170214091529')

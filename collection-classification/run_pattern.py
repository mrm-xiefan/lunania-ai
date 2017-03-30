#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import glob
import logging
import os
import shutil

from keras.backend import tensorflow_backend as tf
from PIL import Image

# app initialize
from app import context
from app.core.log import logging_time
from app.datasets.manager import DatasetStore
from app.models.pattern import (PatternModelContainer,
                                PatternModelContainerAtResNet)
from app.models.manager import ModelStore
from app.prediction.processor import ClassificationProcessor
from app.services.pattern import PatternClassificationService
from app.utils import image_utils
from typing import List

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

LOGGER = logging.getLogger(__name__)


def continue_learning(epochs: List[int],
                      parent_model_key: str,
                      pattern_classification_service: PatternClassificationService):

    model_key = parent_model_key
    for epoch in epochs:
        LOGGER.debug('learn start:%s', model_key)
        model_key = pattern_classification_service.continue_learning(
            model_key,
            '',
            '',
            (300, 200, 3), epoch,
            is_data_aug=True, aug_rate=3)
        tf.clear_session()


def increase():

    org_dir = '/var/ria-dataset/app/train/pattern-crop-20170307-7class'
    save_dir = '/var/ria-dataset/app/train/pattern-crop-20170307-7class-increase'

    image_utils.increase_img_data(org_dir, save_dir, 1100,
                                  generator_option={
                                      'vertical_flip': True,
                                      'horizontal_flip': True,
                                      'width_shift_range': 0.1,
                                      'height_shift_range': 0.1,
                                      'zoom_range': 0.2
                                  },
                                  flow_option={
                                      'target_size': (300, 200)
                                  })


def divide_to_validation():

    org_dir = '/var/ria-dataset/app/train/pattern-crop-20170307-7class-increase'
    divide_dir = '/var/ria-dataset/app/train/pattern-crop-20170307-7class-validation'

    image_utils.divide_to_validation(org_dir, divide_dir, 100)


def copy_from_crop_dir():

    crop_img_dir = '/var/ria-dataset/app/fcn/crop/pattern_8s/org'
    save_dir = '/var/ria-dataset/app/train/pattern-crop-20170307-7class'

    image_utils.copy_from_crop_dir(crop_img_dir, save_dir)


@logging_time
def predict_by_dir(dataset_store: DatasetStore, model_store: ModelStore,
                   model_key: str):

    target_dir = ''
    dist_dir = ''

    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)

    model_container = model_store.load(model_key, use_all_in_one=False)
    model = model_container.get_model()

    label = ['animal', 'floral', 'folklore',
             'geometric', 'logo', 'plaid', 'stripe']
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

    generator = dataset_store.load('',
                                   (300, 200),
                                   generator_option={
                                       'rescale': 1. / 255
                                   }, flow_option={
                                       'shuffle': False
                                   })

    pattern_classification_service = PatternClassificationService(model_store,
                                                                  dataset_store)

    label = ['animal', 'floral', 'folklore',
             'geometric', 'logo', 'plaid', 'stripe']

    processor = ClassificationProcessor(label)
    result = pattern_classification_service.predict(
        model_key,  generator)
    result = processor.process(result)

    with open('./pattern-result.txt', 'w') as f:
        f.writelines(list(map(lambda x: x + '\r\n', result)))


if __name__ == '__main__':
    context.Initializer().initialize()

    model_store = context.get(
        context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore
    dataset_store = context.get(
        context.CONTEXT_KEY_DATASET_STORE, DatasetStore)  # type: DatasetStore

    pattern_classification_service = PatternClassificationService(model_store,
                                                                  dataset_store)

    model_container = PatternModelContainerAtResNet(7)
    new_model_key = pattern_classification_service.learn(model_container,
                                                         model_container.get_model_key() + '-pattern-20170307-7class',
                                                         'pattern-crop-20170307-7class-increase',
                                                         'pattern-crop-20170307-7class-validation',
                                                         (3, 300, 200), 30,
                                                         is_data_aug=True, aug_rate=3)

    tf.clear_session()

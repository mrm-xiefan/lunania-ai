#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import glob
import logging
import os
import random
from typing import Tuple
from pathlib import Path

import numpy as np
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img

from PIL import Image

from app.constants.color_palette import LabelEnum
from app.core.callback import LearningTimeLogger
from app.models import fcn
from app.models.manager import ModelStore
from app.prediction.processor import CrfSegmentationProcessor
from app.utils import layer_utils, image_utils

LOGGER = logging.getLogger(__name__)


class FcnService:

    def __init__(self, model_store: ModelStore):
        self.model_store = model_store

    def learn_continuous(self, learn_data_dir: str, nb_epoch: int,
                         size: Tuple[int], fcn_classes: int,
                         model_key: str,
                         epoch_sum: int, is_override_trainable=True,
                         set_upsample_weights=True):
        '''
        arguments
            size: (height, width)
        '''

        LOGGER.debug('key: %s', model_key)
        model_container = self.model_store.load(
            model_key, use_all_in_one=True)
        model = model_container.get_model()
        LOGGER.debug('model loaded.')

        if set_upsample_weights:
            _set_upsample_weights(model)

        if is_override_trainable:
            is_recompile = False
            for layer in model.layers:
                if 'merge' not in layer.name and not layer.trainable and not is_recompile:
                    is_recompile = True

                layer.trainable = True
            # recomple optimizer

            if is_recompile:
                LOGGER.info('optimizer recompile.')
                fcn.FcnModelContainer(1, 1, 1).compile(model)

        save_dir_name = model_key + str(epoch_sum)

        return self._learn(save_dir_name, model, nb_epoch,
                           learn_data_dir, size[0], size[1], fcn_classes), 'test'

    def learn_by_store_model(self, learn_data_dir: str, nb_epoch: int,
                             img_width: int, img_height: int, fcn_classes: int,
                             key, is_override_trainable=True):

        model = self.model_store.load(key).get_model()
        LOGGER.debug('model loaded.')

        if is_override_trainable:
            for layer in model.layers:
                layer.trainable = True

        return self._learn(self.__class__.__name__, model, nb_epoch,
                           learn_data_dir, img_height, img_width, fcn_classes)

    def learn_using_prev_weights(self, learn_data_dir: str, nb_epoch: int,
                                 img_width: int, img_height: int, fcn_classes: int,
                                 prev_model_key, epoch_sum: int, is_trainable=True,
                                 set_upsample_weights=True):
        '''
        学習済のモデルについて、分類クラス数が異なる入力データを用いて学習させる。
        '''
        fcn_model = fcn.Fcn32sModel(img_width, img_height, fcn_classes)

        model = fcn_model.create_model()
        model = _set_prev_model_weights(
            self.model_store, model, prev_model_key, is_trainable)
        LOGGER.debug('previous model weights set.')

        if set_upsample_weights:
            _set_upsample_weights(model)

        save_dir_name = fcn_model.__class__.__name__ + '-upsample-' + \
            str(fcn_classes) + 'classes-' + str(epoch_sum)

        fcn_model.compile(model)
        return self._learn(save_dir_name, model, nb_epoch,
                           learn_data_dir, img_height, img_width, fcn_classes)

    def learn_using_vgg16_weights(self, learn_data_dir: str, nb_epoch: int,
                                  img_width: int, img_height: int,
                                  fcn_classes: int, set_upsample_weights=True):
        '''
        FCN32sモデルの畳み込み層について、重みの初期値にVGG16の重みをセットする
        VGG16の重みをセットした層は学習をさせない
        '''
        model_container = fcn.Fcn32sModel((img_height, img_width), fcn_classes)

        model = model_container.get_model((1, 2))
        layer_utils.set_vgg16_weights(model, is_trainable=False)
        LOGGER.debug('vgg16 weights set.')

        if set_upsample_weights:
            _set_upsample_weights(model)

        save_dir_name = model_container.get_model_key() + '-vgg16'
        if set_upsample_weights:
            save_dir_name = save_dir_name + '-upsample'
        save_dir_name = save_dir_name + '-' + \
            str(fcn_classes) + 'classes-' + str(nb_epoch)

        model_container.compile(model)
        return self._learn(save_dir_name, model, nb_epoch,
                           learn_data_dir, img_height,
                           img_width, fcn_classes), model_container.get_model_key()

    def learn_using_fcn32s_weights(self, learn_data_dir: str, nb_epoch: int,
                                   img_width: int, img_height: int,
                                   fcn_classes: int, fcn32s_model_key,
                                   is_trainable=True):
        '''
        FCN16sモデルの重みの初期値に、FCN32sで学習した重みをセットして学習する
        '''
        fcn_model = fcn.Fcn16sModel(img_width, img_height, fcn_classes)

        model = fcn_model.create_model()
        model = _set_prev_fcn_weights(
            self.model_store, model, fcn32s_model_key, is_trainable)
        LOGGER.debug('FCN32s weights set.')

        fcn_model.compile(model)
        return self._learn(fcn_model.__class__.__name__, model, nb_epoch,
                           learn_data_dir, img_height, img_width, fcn_classes)

    def learn_using_fcn16s_weights(self, learn_data_dir: str, nb_epoch: int,
                                   img_width: int, img_height: int,
                                   fcn_classes: int, fcn16s_model_key,
                                   is_trainable=True):
        '''
        FCN8sモデルの重みの初期値に、FCN16sで学習した重みをセットして学習する
        '''
        fcn_model = fcn.Fcn8sModel(img_width, img_height, fcn_classes)

        model = fcn_model.create_model()
        model = _set_prev_fcn_weights(
            self.model_store, model, fcn16s_model_key, is_trainable)
        LOGGER.debug('FCN16s weights set.')

        fcn_model.compile(model)
        return self._learn(fcn_model.__class__.__name__, model, nb_epoch,
                           learn_data_dir, img_height, img_width, fcn_classes), fcn_model.__class__.__name__

    def learn_resnet(self, learn_data_dir: str, nb_epoch: int,
                     img_width: int, img_height: int,
                     fcn_classes: int):
        '''
        resnetモデルを学習する
        '''
        fcn_model = fcn.Resnet50Model((img_height, img_width), fcn_classes)

        model = fcn_model.create_model()

        fcn_model.compile(model)

        return self._learn(fcn_model.__class__.__name__, model, nb_epoch,
                           learn_data_dir, img_height, img_width, fcn_classes), fcn_model.__class__.__name__

    def _learn(self, save_model_key: str, model: Model, nb_epoch: int,
               learn_data_dir, img_height: int, img_width: int,
               fcn_classes: int, print_summary=True) -> str:
        LOGGER.debug('learn start:%s', model.__class__.__name__)

        if print_summary:
            model.summary()

        sample_count, generator = _create_learn_data_generator(
            learn_data_dir, img_height, img_width, fcn_classes)

        # set default callbacks
        lt_cb = LearningTimeLogger(is_log_batch=False)
        history = model.fit_generator(generator(), samples_per_epoch=sample_count,
                                      nb_epoch=nb_epoch, verbose=1, callbacks=[lt_cb])

        model_key = self.model_store.save(
            save_model_key, model, history, fit_callbacks=[lt_cb])

        return model_key

    def predict(self, key, img_path: str, size: Tuple[int],
                fcn_classes: int,
                print_summary=True):
        '''
        arguments
            size: (height, width)
        '''

        model = self.model_store.load(key, use_all_in_one=False).get_model()

        LOGGER.debug('Model loaded.')

        if print_summary:
            model.summary()

        img = self.preprocces_for_predict(img_path, size)
        result = model.predict_on_batch(img)

        _org = np.copy(np.asarray(
            load_img(img_path, grayscale=False, target_size=size)))
        return [fcn.SegmentationProcessor(size).process(result),
                CrfSegmentationProcessor(size, fcn_classes).process(result, _org)]

    def preprocces_for_predict(self, img_path: str, size: Tuple[int]) -> np.ndarray:
        '''
        argments
            img_path: image_path
            size: (height, width)
        '''
        img = load_img(img_path, grayscale=False,
                       target_size=size)
        return image_utils.to_array_and_reshape(img, samplewise=True)


def _set_upsample_weights(model):
    for layer in model.layers:
        if 'deconv' in layer.name:
            layer_utils.interp_upsample_filter(layer, verbose=True)
            # TODO test code.
            layer.trainable = False
            LOGGER.debug('Upsampling weights set in %s layer ', layer.name)
    return model


def _set_prev_fcn_weights(model_store: ModelStore, model, prev_model_key, is_trainable: bool):
    model_store.load_weights(model, prev_model_key)

    if not is_trainable:
        for layer in model.layers:
            if 'branch' not in layer.name and 'fcn' not in layer.name:
                layer.trainable = is_trainable
                LOGGER.debug('%s layer trainable: %s' %
                             (layer.name, str(is_trainable)))

    return model


def _set_prev_model_weights(model_store: ModelStore, model, prev_model_key, is_trainable: bool):
    '''
    前回のモデルで学習した重みをレイヤ名称にて紐付け、重みを設定する。
    '''
    prev_model = model_store.load(prev_model_key).get_model()

    for layer in model.layers:
        prev_layer = prev_model.get_layer(name=layer.name)

        if prev_layer is not None:
            model_weights = layer.get_weights()
            prev_weights = prev_layer.get_weights()

            if len(model_weights) > 0 and model_weights[0].shape == prev_weights[0].shape:
                layer.set_weights(prev_weights)
                layer.trainable = is_trainable
                LOGGER.debug(
                    'set previous model weights in %s layer', layer.name)
    return model


def _create_learn_data_generator(dir_path: str,
                                 img_height: int, img_width: int,
                                 fcn_classes: int):

    LOGGER.debug('dir_path:%s', dir_path)
    LOGGER.debug('glob:%s', os.path.join(dir_path, '*.jpg'))
    files = list(map(os.path.abspath, glob.glob(
        os.path.join(dir_path, '*.jpg'))))
    file_count = len(files)
    LOGGER.debug('file count:%s', file_count)

    def _generator():
        while 1:
            random.shuffle(files)
            for file in files:
                file_name, ext = os.path.splitext(file)

                train_image = load_img(file_name + ext, grayscale=False,
                                       target_size=(img_height, img_width))

                image = img_to_array(train_image) / 255
                # samplewise_center. ref:keras\preprocessing\image.py,
                image -= np.mean(image, axis=2, keepdims=True)
                image = image.reshape((1,) + image.shape)

                target_image = np.load(file_name + '.npy')
                label_image = np.zeros([img_height, img_width, fcn_classes])

                for i in range(img_height):
                    for j in range(img_width):
                        if fcn_classes == 25:
                            label_image[i, j, int(target_image[i][j])] = 1
                        elif fcn_classes == 56:
                            label_image[i, j, int(target_image[i][j] - 1)] = 1
                        else:
                            label_image[i, j, int(target_image[i][j])] = 1

                label_image = label_image.reshape(
                    (1, img_height * img_width, fcn_classes))

                yield (image, label_image)

    return [file_count, _generator]

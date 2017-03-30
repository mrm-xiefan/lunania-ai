#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import glob
import logging
import os
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
from keras import backend as K
from keras.layers import (Activation, Convolution2D, Cropping2D,
                          Deconvolution2D, Dropout, Input, MaxPooling2D,
                          Permute, Reshape, UpSampling2D, ZeroPadding2D, merge)
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50

from PIL import Image
from app.constants.color_palette import Label
from app.models.manager import ModelStore
from app.prediction.processor import PredictionResultProcessor
from typing import Tuple

from . import ModelContainer

LOGGER = logging.getLogger(__name__)


class FcnModelContainer(ModelContainer):

    def __init__(self, img_width: int, img_height: int, fcn_classes: int):
        self.img_width = img_width
        self.img_height = img_height
        self.fcn_classes = fcn_classes

    def compile(self, model: Model):
        model.compile(loss="categorical_crossentropy",
                      optimizer='Adadelta',
                      metrics=["accuracy"])


class Fcn8sModel(FcnModelContainer):
    '''
    FCN8sのモデルを表すクラス
    '''

    def __init__(self, img_width: int, img_height: int, fcn_classes: int):
        super().__init__(img_width, img_height, fcn_classes)

    def create_model(self):

        _img_width = self.img_width
        _img_height = self.img_height
        _fcn_classes = self.fcn_classes

        input_shape = (_img_height, _img_width, 3)
        input_img = Input(shape=input_shape, name='input_1')
        # (600*400*3)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_1')(input_img)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_1')(x)
        # (300*200*64)

        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_1')(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_2')(x)
        # (150*100*128)

        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_1')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_2')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_3')(x)
        # (75*50*256)

        # split layer
        p3 = x
        p3 = Convolution2D(_fcn_classes, 1, 1,
                           activation='relu', name='branch_conv_3')(p3)
        # (75*50*56)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_4')(x)
        # (37*25*512)

        # split layer
        p4 = x
        p4 = Convolution2D(_fcn_classes, 1, 1,
                           activation='relu', name='branch_conv_4')(p4)

        p4 = Deconvolution2D(_fcn_classes, 4, 4,
                             output_shape=(1, 77, 52, _fcn_classes),
                             subsample=(2, 2),
                             border_mode='valid',
                             name='branch_deconv_4')(p4)
        p4 = Cropping2D(cropping=((1, 1), (1, 1)), name='branch_crop_4')(p4)
        # (75*50*56)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_5')(x)
        # (18*12*512)

        x = Convolution2D(4096, 7, 7, activation='relu',
                          border_mode='same', name='convolution2d_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Convolution2D(4096, 1, 1, activation='relu',
                          border_mode='same', name='convolution2d_2')(x)
        x = Dropout(0.5, name='dropout_2')(x)

        p5 = x
        p5 = Convolution2D(_fcn_classes, 1, 1,
                           activation='relu', name='branch_conv_5')(p5)
        p5 = Deconvolution2D(_fcn_classes, 8, 8,
                             output_shape=(1, 79, 54, _fcn_classes),
                             subsample=(4, 4),
                             border_mode='valid',
                             name='branch_deconv_5')(p5)
        p5 = Cropping2D(cropping=((2, 2), (2, 2)), name='branch_crop_5')(p5)
        # (75*50*56)

        # merge scores
        merged = merge([p3, p4, p5], mode='sum', name='fcn8s_merge')
        x = Deconvolution2D(_fcn_classes, 16, 16,
                            output_shape=(1, 608, 408, _fcn_classes),
                            subsample=(8, 8),
                            border_mode='valid',
                            name='fcn8s_deconv_1')(merged)
        x = Cropping2D(cropping=((4, 4), (4, 4)), name='fcn8s_crop_1')(x)

        x = Reshape((_img_width * _img_height, _fcn_classes),
                    name='reshape_1')(x)
        out = Activation("softmax", name='fcn8s_activation_1')(x)
        # (600,400,56)
        return Model(input_img, out)

    def get_model(self):
        return self.create_model()


class Fcn16sModel(FcnModelContainer):
    '''
    FCN16sのモデルを表すクラス
    '''

    def __init__(self, img_width: int, img_height: int, fcn_classes: int):
        super().__init__(img_width, img_height, fcn_classes)

    def create_model(self):

        _img_width = self.img_width
        _img_height = self.img_height
        _fcn_classes = self.fcn_classes

        input_shape = (_img_height, _img_width, 3)
        input_img = Input(shape=input_shape, name='input_1')
        #(600*400*3)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_1')(input_img)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_1')(x)
        #(300*200*64)

        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_1')(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_2')(x)
        #(150*100*128)

        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_1')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_2')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_3')(x)
        #(75*50*256)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_4')(x)
        # (37*25*512)

        p4 = x
        p4 = Convolution2D(_fcn_classes, 1, 1, activation='relu',
                           border_mode='same', name='branch_conv_4')(p4)
        # (37*25*56)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_5')(x)
        # (18*12*512)

        x = Convolution2D(4096, 7, 7, activation='relu',
                          border_mode='same', name='convolution2d_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Convolution2D(4096, 1, 1, activation='relu',
                          border_mode='same', name='convolution2d_2')(x)
        x = Dropout(0.5, name='dropout_2')(x)
        x = Convolution2D(_fcn_classes, 1, 1,
                          activation='relu', border_mode='same', name='convolution2d_3')(x)
        x = Deconvolution2D(_fcn_classes, 4, 4,
                            output_shape=(1, 35, 23, _fcn_classes),
                            subsample=(2, 2),
                            border_mode='same',
                            name='fcn16s_deconv_1')(x)
        x = ZeroPadding2D((1, 1), name='fcn16s_zero_1')(x)
        # (37*25*56)

        merged = merge([p4, x], mode='sum', name='fcn16s_merge')

        x = Deconvolution2D(_fcn_classes, 64, 64,
                            output_shape=(1, 612, 408, _fcn_classes),
                            subsample=(15, 14),
                            border_mode='valid',
                            name='fcn16s_deconv_2')(merged)
        # x = ZeroPadding2D((4, 0))(x)
        x = Cropping2D(cropping=((6, 6), (4, 4)), name='fcn16s_crop_1')(x)

        x = Reshape((_img_width * _img_height, _fcn_classes),
                    name='reshape_1')(x)
        out = Activation("softmax", name='fcn16s_activation_1')(x)
        # (600,400,56)
        model = Model(input_img, out)

        LOGGER.debug('model created.')

        return model

    def get_model(self, input_shape):
        return self.create_model(input_shape)


class Fcn32sModel(FcnModelContainer):
    '''
    FCN32sのモデルを表すクラス
    '''

    def __init__(self, size: Tuple[int], fcn_classes: int):
        '''
        arguments
            size: (height, width)
        '''
        super().__init__(size[1], size[0], fcn_classes)

    def create_model(self):
        _img_width = self.img_width
        _img_height = self.img_height
        _fcn_classes = self.fcn_classes

        input_shape = (_img_height, _img_width, 3)

        input_img = Input(shape=input_shape, name='input_1')
        #(600*400*3)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_1')(input_img)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_1')(x)
        #(300*200*64)

        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_1')(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_2')(x)
        #(150*100*128)

        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_1')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_2')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_3')(x)
        #(75*50*256)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_4')(x)
        # (37*25*512)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_5')(x)
        # (18*12*512)

        x = Convolution2D(4096, 7, 7, activation='relu',
                          border_mode='same', name='convolution2d_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Convolution2D(4096, 1, 1, activation='relu',
                          border_mode='same', name='convolution2d_2')(x)
        x = Dropout(0.5, name='dropout_2')(x)
        x = Convolution2D(_fcn_classes, 1, 1,
                          activation='relu', border_mode='same', name='convolution2d_3')(x)

        x = Deconvolution2D(_fcn_classes, 64, 64,
                            output_shape=(None, 612, 408, _fcn_classes),
                            subsample=(34, 34),
                            border_mode='same',
                            name='deconvolution2d_1')(x)
        x = Cropping2D(cropping=((6, 6), (4, 4)), name='cropping2d_1')(x)

        x = Reshape((_img_width * _img_height, _fcn_classes),
                    name='reshape_1')(x)
        out = Activation("softmax", name='activation_1')(x)
        # (600,400,56)
        model = Model(input_img, out)

        LOGGER.debug('model created.')

        return model

    def get_model(self, input_shape):
        return self.create_model()


class Resnet50Model(FcnModelContainer):
    '''
    50層のResnetを用いたモデルを表すクラス
    '''

    def __init__(self, size: Tuple[int], fcn_classes: int):
        '''
        arguments
            size: (height, width)
        '''
        super().__init__(size[1], size[0], fcn_classes)

    def create_model(self):
        _img_width = self.img_width
        _img_height = self.img_height
        _fcn_classes = self.fcn_classes

        input_shape = (_img_height, _img_width, 3)

        input_img = Input(shape=input_shape, name='input_1')

        model = ResNet50(include_top=False, input_tensor=input_img)
        x = model.layers[-2].output
        LOGGER.debug(type(x))

        # FC layer
        x = Cropping2D(cropping=((0, 1), (0, 1)), name='cropping2d_2')(x)
        x = Deconvolution2D(_fcn_classes, 64, 64,
                            output_shape=(None, 612, 408, _fcn_classes),
                            subsample=(34, 34),
                            border_mode='same',
                            name='deconvolution2d_1')(x)
        x = Cropping2D(cropping=((6, 6), (4, 4)), name='cropping2d_1')(x)

        x = Reshape((_img_width * _img_height, _fcn_classes),
                    name='reshape_1')(x)
        out = Activation("softmax", name='activation_fc')(x)
        model = Model(model.layers[0].input, out)

        LOGGER.debug('model created.')

        return model

    def get_model(self, input_shape):
        return self.create_model()


class Fcn32sUpsampleModel(FcnModelContainer):
    '''
    FCN32sのモデルを表すクラス
    UpSampling2D層を使用
    '''

    def __init__(self, img_width: int, img_height: int, fcn_classes: int):
        super().__init__(img_width, img_height, fcn_classes)

    def create_model(self):
        _img_width = self.img_width
        _img_height = self.img_height
        _fcn_classes = self.fcn_classes

        input_shape = (_img_height, _img_width, 3)

        input_img = Input(shape=input_shape, name='input_1')
        #(600*400*3)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_1')(input_img)
        x = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same', name='conv_1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_1')(x)
        #(300*200*64)

        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_1')(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same', name='conv_2_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_2')(x)
        #(150*100*128)

        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_1')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_2')(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same', name='conv_3_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_3')(x)
        #(75*50*256)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_4_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_4')(x)
        # (37*25*512)

        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_1')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_2')(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same', name='conv_5_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_5')(x)
        # (18*12*512)

        x = Convolution2D(4096, 7, 7, activation='relu',
                          border_mode='same', name='convolution2d_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Convolution2D(4096, 1, 1, activation='relu',
                          border_mode='same', name='convolution2d_2')(x)
        x = Dropout(0.5, name='dropout_2')(x)
        x = Convolution2D(_fcn_classes, 1, 1,
                          activation='relu', border_mode='same', name='convolution2d_3')(x)

        x = UpSampling2D(size=(34, 34), name='upsampling2d_1')(x)
        x = Cropping2D(cropping=((6, 6), (4, 4)), name='cropping2d_1')(x)

        x = Reshape((_fcn_classes, _img_width * _img_height),
                    name='reshape_1')(x)
        x = Permute((2, 1), name='permute_1')(x)
        out = Activation("softmax", name='activation_1')(x)
        # (600,400,56)
        model = Model(input_img, out)

        LOGGER.debug('model created.')

        return model

    def get_model(self, input_shape):
        return self.create_model()


class SegmentationProcessor(PredictionResultProcessor[np.ndarray, np.ndarray]):
    '''
    FCNによるpredict結果を処理するためのクラス
    '''

    def __init__(self, size: Tuple[int]):
        '''
        argments
            size: (height, width)
        '''
        self._size = size

    def process(self, result: np.ndarray) -> np.ndarray:
        label_data = []

        for i in range(len(result[0])):
            label_data.append(np.argmax(result[0][i]))

        # reshape to original size
        return np.reshape(label_data, self._size)

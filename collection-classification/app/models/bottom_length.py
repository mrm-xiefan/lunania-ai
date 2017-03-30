#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging

from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop

from app.utils.resnet_utils import ResnetBuilder

from . import ModelContainer

LOGGER = logging.getLogger(__name__)


class BottomLengthModelContainer(ModelContainer):

    def __init__(self):
        super().__init__()
        self.feature_layer_count = 0

    def get_feature_model(self, input_shape):

        input = Input(shape=input_shape, name='input_1')
        x = ZeroPadding2D((1, 1))(input)

        x = Convolution2D(64, 3, 3, activation='relu',
                          name='conv1_1', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(64, 3, 3, activation='relu',
                          name='conv1_2', trainable=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          name='conv2_1', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, 3, 3, activation='relu',
                          name='conv2_2', trainable=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          name='conv3_1', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          name='conv3_2', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, 3, 3, activation='relu',
                          name='conv3_3', trainable=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv4_1', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv4_2', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv4_3', trainable=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv5_1', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv5_2', trainable=False)(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, 3, 3, activation='relu',
                          name='conv5_3', trainable=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # update vgg16 model layer count
        self.feature_layer_count = 45

        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(9, activation='softmax')(x)

        return Model(input, output)

    def compile(self, model: Model) -> None:
        rmsprop = RMSprop(lr=0.001)
        model.compile(optimizer=rmsprop, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def get_model(self, input_shape):
        return self.get_feature_model(input_shape)

    def get_model_key(self) -> str:
        return self.__class__.__name__


class BottomLengthModelContainerAtResNet(BottomLengthModelContainer):

    def __init__(self, classes: int):
        self._classes = classes
        super().__init__()

    def get_feature_model(self, input_shape):

        return ResnetBuilder.build_resnet_34(input_shape, self._classes)

    def compile(self, model: Model) -> None:
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

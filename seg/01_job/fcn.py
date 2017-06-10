#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import os
from os.path import join as join_path
import glob
import random
import utils
import config
import traceback
import logging.config
from datetime import datetime
from luna import LunaExcepion

import h5py
import numpy as np
from keras import backend as K
from keras.layers import (Activation, Conv2D, Cropping2D,
                          Conv2DTranspose, Dropout, Input, MaxPooling2D,
                          Permute, Reshape, UpSampling2D, ZeroPadding2D, merge)
from keras.models import Model
from keras.utils import vis_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

class Fcn:

    MODEL_FILE_NAME = 'model.yaml'
    VISUALIZED_MODEL_FILE_NAME = 'model.png'
    WEIGHTS_FILE_NAME = 'model.h5'
    ALL_IN_MODEL_FILE_NAME = 'model_all.h5'

    def __init__(self, model_type):
        self.model_type = model_type
        self.epoch = 1
        if (self.model_type == 'vgg'):
            self.model = createFCN32sModel()
            logger.info("model created")
            set_vgg16_weights(self.model)
            logger.info("vgg16 weights seted")
            self.compileModel()
            logger.info("model compiled")
        elif (self.model_type == 'resnet'):
            self.model = createResNetModel()
            logger.info("model created")
            self.compileModel()
            logger.info("model compiled")

    def compileModel(self):
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer='Adadelta',
            metrics=["accuracy"]
        )

    def train(self, dataset, epoches):
        sample_count, generator = createDataGenerater(dataset)
        history = self.model.fit_generator(
            generator(),
            steps_per_epoch=sample_count,
            epochs=epoches,
            verbose=1
        )

        save_dir = join_path(config.model_dir, self.model_type + '-' + str(config.classes) + 'class-' + str(self.epoch) + 'epoch-' + datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        logger.info('model path: %s', save_dir)
        with open(os.path.join(save_dir, self.MODEL_FILE_NAME), mode='w', encoding='utf-8') as model_file:
            model_file.write(self.model.to_yaml())
        vis_utils.plot_model(self.model, to_file=os.path.join(save_dir, self.VISUALIZED_MODEL_FILE_NAME), show_shapes=True)
        self.model.save_weights(os.path.join(save_dir, self.WEIGHTS_FILE_NAME))
        self.model.save(os.path.join(save_dir, self.ALL_IN_MODEL_FILE_NAME))


def createDataGenerater(dataset):

    jpg_path = join_path(dataset, '*.jpg')
    files = list(map(os.path.abspath, glob.glob(jpg_path)))
    file_count = len(files)
    logger.debug('file count: %s', file_count)

    def generator():

        while 1:
            random.shuffle(files)
            for file in files:
                file_name, ext = os.path.splitext(file)

                train_image = load_img(file_name + ext, grayscale=False,
                                       target_size=(config.img_height, config.img_width))

                image = img_to_array(train_image) / 255
                image = image.reshape((1,) + image.shape)

                target_image = np.load(file_name + '.npy')
                label_image = np.zeros([config.img_height, config.img_width, config.classes])

                for i in range(config.img_height):
                    for j in range(config.img_width):
                        if target_image[i][j] == 255:
                            label_image[i, j, config.classes - 1] = 1
                        else:
                            label_image[i, j, int(target_image[i][j])] = 1

                label_image = label_image.reshape(
                    (1, config.img_height * config.img_width, config.classes))

                yield (image, label_image)

    return [file_count, generator]

def createFCN32sModel():

    input_shape = (config.img_height, config.img_width, 3)

    input_img = Input(shape=input_shape, name='input_1')
    #(500*500*3)
    x = Conv2D(64, (3, 3), activation='relu',
                        padding='same', name='conv_1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu',
                        padding='same', name='conv_1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_1')(x)
    #(250*250*64)

    x = Conv2D(128, (3, 3), activation='relu',
                        padding='same', name='conv_2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
                        padding='same', name='conv_2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_2')(x)
    #(125*125*128)

    x = Conv2D(256, (3, 3), activation='relu',
                        padding='same', name='conv_3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
                        padding='same', name='conv_3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
                        padding='same', name='conv_3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_3')(x)
    #(62*62*256)

    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_4')(x)
    # (31*31*512)

    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
                        padding='same', name='conv_5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2d_5')(x)
    # (15*15*512)

    x = Conv2D(4096, (7, 7), activation='relu',
                        padding='same', name='convolution2d_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Conv2D(4096, (1, 1), activation='relu',
                        padding='same', name='convolution2d_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = Conv2D(config.classes, (1, 1),
                        activation='relu', padding='same', name='convolution2d_3')(x)

    h_grid = int(round(config.img_height / 100))
    w_grid = int(round(config.img_width / 100))
    x = Conv2DTranspose(config.classes, (64, 64),
                        output_shape=(None, h_grid * 102, w_grid * 102, config.classes),
                        strides=(34, 34),
                        padding='same',
                        name='deconvolution2d_1')(x)
    x = Cropping2D(cropping=((h_grid, h_grid), (w_grid, w_grid)), name='cropping2d_1')(x)

    x = Reshape((config.img_width * config.img_height, config.classes),
                name='reshape_1')(x)
    out = Activation("softmax", name='activation_1')(x)
    # (500,500,56)
    model = Model(input_img, out)

    return model

def set_vgg16_weights(model):

    with h5py.File(config.vgg16_weights_file) as weight_file:
        generator = conv_weights_generator(weight_file)

        count = 0
        for layer in model.layers:
            if count > 12:
                break

            if 'conv' in layer.name:
                weights = generator.__next__()
                layer.set_weights(weights)

                layer.trainable = False

                count += 1

def conv_weights_generator(file):
    while 1:
        for k in range(file.attrs['nb_layers']):
            g = file['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])]
            if len(weights) > 0:
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
                yield(weights)

def createResNetModel():
    
    input_shape = (config.img_height, config.img_width, 3)

    input_img = Input(shape=input_shape, name='input_1')

    model = ResNet50(include_top=False, input_tensor=input_img)
    x = model.layers[-2].output

    # FC layer
    x = Cropping2D(cropping=((0, 1), (0, 1)), name='cropping2d_2')(x)
    h_grid = int(round(config.img_height / 100))
    w_grid = int(round(config.img_width / 100))
    x = Conv2DTranspose(config.classes, (64, 64),
        output_shape=(None, h_grid * 102, w_grid * 102, config.classes),
        strides=(34, 34),
        padding='same',
        name='deconvolution2d_1')(x)
    x = Cropping2D(cropping=((h_grid, h_grid), (w_grid, w_grid)), name='cropping2d_1')(x)

    x = Reshape((config.img_width * config.img_height, config.classes), name='reshape_1')(x)
    out = Activation("softmax", name='activation_fc')(x)
    model = Model(model.layers[0].input, out)

    return model


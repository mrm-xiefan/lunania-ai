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
from pathlib import Path
from luna import LunaExcepion

import h5py
import numpy as np
from keras import backend as K
from keras.layers import (Activation, Conv2D, Cropping2D,
                          Conv2DTranspose, Dropout, Input, MaxPooling2D,
                          Permute, Reshape, UpSampling2D, ZeroPadding2D, merge)
from keras import models
from keras.models import Model
from keras.utils import vis_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50
from PIL import Image

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

class Fcn:

    MODEL_FILE_NAME = 'model.yaml'
    VISUALIZED_MODEL_FILE_NAME = 'model.png'
    WEIGHTS_FILE_NAME = 'model.h5'
    ALL_IN_MODEL_FILE_NAME = 'model_all.h5'

    def __init__(self, model_type = 'predict'):
        self.model_type = model_type

    def createModel(self):
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

    def load(self, model_name):
        model_path = os.path.join(config.model_dir, model_name, self.ALL_IN_MODEL_FILE_NAME)
        logger.debug('model_path: %s', model_path)
        exists_all_in_one = os.path.exists(model_path)
        if exists_all_in_one:
            logger.debug('use all in one.')
            self.model = models.load_model(model_path, custom_objects=None)
        else:
            model_path = os.path.join(config.model_dir, model_name, self.MODEL_FILE_NAME)
            with open(model_path, 'r') as yaml_string:
                self.model = models.model_from_yaml(yaml_string, custom_objects=None)
            self.model.load_weights(os.path.join(config.model_dir, model_name, self.WEIGHTS_FILE_NAME), by_name=True)

    def predict(self, image_file):
        pre_array = np.asarray(Image.open(image_file))
        logger.debug(pre_array.shape)
        trim_h = pre_array.shape[0]
        trim_w = pre_array.shape[1]

        img_array = np.zeros([config.img_height, config.img_width, 3])
        img_array[0:pre_array.shape[0], 0:pre_array.shape[1], 0:pre_array.shape[2]] = pre_array
        img_array /= 255
        logger.debug(img_array.shape)

        result = self.model.predict_on_batch(img_array.reshape((1,) + img_array.shape))
        logger.debug(result.shape)

        fresult = getResult(result)
        #img_copy = np.copy(np.uint8(img_array))
        #fresult = getCRFResult(result, img_copy)

        # trim back
        trim_jpg = np.zeros([trim_h, trim_w])
        trim_jpg = fresult[0:trim_h, 0:trim_w]

        trim_jpg, predict_labels = colorful(trim_jpg)

        image_file = Path(image_file)
        file_name = image_file.stem

        im = Image.fromarray(np.uint8(trim_jpg))
        result_img = join_path(config.predict_dir, file_name + '-predict.jpg')
        im.save(result_img)
        return file_name + '-predict.jpg', predict_labels


def getCRFResult(result, img):

    _d = dcrf.DenseCRF2D(config.img_height, config.img_width, config.classes)
    result = result[0]
    label = result.reshape((config.img_height, config.img_width, config.classes)).transpose((2, 0, 1))
    U = unary_from_softmax(label)
    _d.setUnaryEnergy(U)
    _d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    _d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13),
        rgbim=img,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    Q = _d.inference(1)
    return np.argmax(Q, axis=0).reshape((config.img_height, config.img_width))

def getResult(result):

    label_data = []
    for i in range(len(result[0])):
        label_data.append(np.argmax(result[0][i]))
    return np.reshape(label_data, (config.img_height, config.img_width))

def colorfulA(img_array):
    background = np.asarray([0, 0, 0])  # 0
    city = np.asarray([255,40,0])  # 1
    farm = np.asarray([250,245,0])  # 2
    forest = np.asarray([53,161,107])  # 3

    label_colors = np.array([background, city, farm, forest])
    labels = ["background", "city", "farm", "forest"]
    
    rgb_image = np.zeros((img_array.shape[0], img_array.shape[1], 3))
    predict_labels = []
    for h, tmp in enumerate(img_array):
        for w, label in enumerate(tmp):
            rgb_image[h, w] = label_colors[int(label)]
            if int(label) == 0:
                continue
            has_label = False
            for l in predict_labels:
                if l == labels[int(label)]:
                    has_label = True
                    break
            if has_label == False:
                predict_labels.append(labels[int(label)])

    logger.debug('predict_labels: %s', predict_labels)
    img = Image.fromarray(np.uint8(rgb_image))

    return img, predict_labels

def colorful(img_array):

    background = np.asarray([0, 0, 0])  # 0
    aeroplane = np.asarray([255,40,0])  # 1
    bicycle = np.asarray([250,245,0])  # 2
    bird = np.asarray([53,161,107])  # 3
    boat = np.asarray([0,135,60])  # 4
    bottle = np.asarray([0,65,255])  # 5
    bus = np.asarray([102,204,255])  # 6
    car = np.asarray([255,153,160])  # 7
    cat = np.asarray([255,40,255])  # 8
    chair = np.asarray([255,153,0])  # 9
    cow = np.asarray([154,0,121])  # 10
    diningtable = np.asarray([190,0,68])  # 11
    dog = np.asarray([24,24,120])  # 12
    horse = np.asarray([102,51,0])  # 13
    motorbike = np.asarray([100,100,100])  # 14
    person = np.asarray([239,132,92])  # 15
    pottedplant = np.asarray([245,176,144])  # 16
    sheep = np.asarray([255,246,127])  # 17
    sofa = np.asarray([105,189,131])  # 18
    train = np.asarray([165,154,202])  # 19
    tvmonitor = np.asarray([190,190,190])  # 20
    boundary = np.asarray([255, 255, 255])  # 255 => 21

    label_colors = np.array([
        background, aeroplane, bicycle, bird,
        boat, bottle, bus, car, cat, chair,
        cow, diningtable, dog, horse, motorbike,
        person, pottedplant, sheep, sofa,
        train, tvmonitor, boundary
    ])
    labels = [
        "background", "aeroplane", "bicycle", "bird",
        "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor", "boundary"
    ]

    rgb_image = np.zeros((img_array.shape[0], img_array.shape[1], 3))
    predict_labels = []
    for h, tmp in enumerate(img_array):
        for w, label in enumerate(tmp):
            rgb_image[h, w] = label_colors[int(label)]
            if int(label) == 0 or int(label) == 21:
                continue
            has_label = False
            for l in predict_labels:
                if l == labels[int(label)]:
                    has_label = True
                    break
            if has_label == False:
                predict_labels.append(labels[int(label)])

    logger.debug('predict_labels: %s', predict_labels)
    img = Image.fromarray(np.uint8(rgb_image))

    return img, predict_labels

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
                        #output_shape=(None, h_grid * 102, w_grid * 102, config.classes),
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
        #output_shape=(None, h_grid * 102, w_grid * 102, config.classes),
        strides=(34, 34),
        padding='same',
        name='deconvolution2d_1')(x)
    x = Cropping2D(cropping=((h_grid, h_grid), (w_grid, w_grid)), name='cropping2d_1')(x)

    x = Reshape((config.img_width * config.img_height, config.classes), name='reshape_1')(x)
    out = Activation("softmax", name='activation_fc')(x)
    model = Model(model.layers[0].input, out)

    return model


#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

import root

# path to the model weights file.
WEIGHTS_PATH =  root.get_project_root() + '/data/vgg16_weights.h5'
TOP_MODEL_WEIGHTS_PATH = root.get_project_root() + '/data/bottleneck_fc_model.h5'
# model filename
MODEL_FILENAME = 'cnn_model.json'

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT = 150, 150

def get_bottoleneck_model():
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    weight_file = h5py.File(WEIGHTS_PATH)
    for k in range(weight_file.attrs['nb_layers']):

        if k >= len(model.layers):
           # we don't look at the last (fully-connected) layers in the savefile
            break
        group = weight_file['layer_{}'.format(k)]
        weights = [group['param_{}'.format(p)]
                   for p in range(group.attrs['nb_params'])]

        # transform to tensorflow format from theano format.
        for i, weight in enumerate(weights):
            print('loop:' + str(i))
            print(weight)
            if i == 0:
                weights[i] = np.transpose(weights[i], (2, 3, 1, 0))
        model.layers[k].set_weights(weights)
    weight_file.close()

    model.to_json()
    return model

def get_top_model():
    model = Sequential()
    model.add(Flatten(input_shape=(4, 4, 512)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(TOP_MODEL_WEIGHTS_PATH)
    return model

bottle_model = get_bottoleneck_model()
top_model = get_top_model()

print('Model loaded.')

# print model summary
bottle_model.summary()
top_model.summary()

top_model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])


for i in range(0, 10):
    if i % 2 == 0:
        img_src = root.get_project_root() + '/data/train/cat/cat.' + str(i) + '.jpg'
    else:
        img_src = root.get_project_root() +'/data/train/dog/dog.' + str(i) + '.jpg'

    eval_image = load_img(img_src).resize((IMG_WIDTH, IMG_HEIGHT))
    eval_image = img_to_array(eval_image) /255
    result = bottle_model.predict(np.expand_dims(eval_image, axis=0), verbose=0)

    clazz = top_model.predict_classes(result, verbose=0)
    predict = top_model.predict_proba(result, verbose=0)
    print('cat' if clazz[0] == 0 else 'dog', 'src:' + img_src, 'predict:', predict)

print('----end process----')


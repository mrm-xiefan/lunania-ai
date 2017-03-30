#!/usr/bin/env python35
# coding: UTF-8

import os
import numpy as np
import h5py
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.engine.topology import Merge

# path to the model weights file.
WEIGHTS_PATH = '../h5py/vgg16_weights.h5'
TOP_MODEL_WEIGHTS_PATH = '../h5py/top_model_test_3.h5'
# model filename
MODEL_FILENAME = '../model/cnn_model_test_3.json'

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT = 150, 150

def get_bottleneck_model():
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
    return model

def get_top_model():
    model = Sequential()
    model.add(Flatten(input_shape=(4, 4, 512)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    model.load_weights(TOP_MODEL_WEIGHTS_PATH)
    return model

bottle_model = get_bottleneck_model()
top_model = get_top_model()

print('Model loaded.')

# print model summary
bottle_model.summary()
top_model.summary()

top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

dirlist = os.listdir('../data')
dirlist.sort()
classes = {}
print(dirlist)
#for i, cls in enumerate(dirlist):
#    print('file' + str(i) + ':' + cls )
#    classes[i] = cls 


img_src = '../data/dress/_AG10818.jpg'

eval_image = load_img(img_src).resize((IMG_WIDTH, IMG_HEIGHT))
eval_image = img_to_array(eval_image) /255
result = bottle_model.predict(np.expand_dims(eval_image, axis=0), verbose=0)

clazz = top_model.predict_classes(result, verbose=0)
predict = top_model.predict_proba(result, verbose=0)


for i in range(len(predict[0])):
    formatPredict = "{0:.5f}".format(predict[0][i] * 100)
    print(dirlist[i] + ': ' + formatPredict + '%')

#print(predict)
print(clazz)

print('----end process----')


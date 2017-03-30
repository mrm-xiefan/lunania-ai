#!/usr/bin/env python35
# coding: UTF-8

import os
import logging
import logging.config

import yaml

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as  K
from keras.utils.np_utils import convert_kernel
from keras.utils import np_utils
import tensorflow as tf

# force update cwd.
os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
# initialize logging configuration.
with open('src/logging.yaml', 'r') as file:
    log_config = yaml.load(file)
logging.config.dictConfig(log_config)
print('init end.')

LOGGER = logging.getLogger(__name__)

LOGGER.debug('learning start.')

# 読み込むモデルの重みファイル、書き込むモデルの重みファイルのパスを指定
WEIGHTS_PATH = 'h5py/vgg16_weights.h5'
TOP_MODEL_WEIGHTS_PATH = 'h5py/top_model_test_3.h5'

# 画像サイズを指定
IMG_WIDTH, IMG_HEIGHT = 150, 150

DATA_DIR = 'data/item'
NB_SAMPLES = 3192

# 学習回数を指定
NB_EPOCH = 50


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # VGG16モデルの重みを読み込み(ImageNetを学習済)
    assert os.path.exists(WEIGHTS_PATH), 'Model weights not found (see "WEIGHTS_PATH" variable in script).'
    f = h5py.File(WEIGHTS_PATH)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        # TensorFlowの形式に重みファイルを変換
        if len(weights) > 0:
             weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

        model.layers[k].set_weights(weights)
    f.close()
    LOGGER.debug('Model loaded.')

    # データを読み込ませる
	# directory：ディレクトリへのパス。分類ごとのサブディレクトリを含み、
    #            そのサブディレクトリにPNGかJPG形式の画像が含まれていなければならない。
	# target_size：画像のサイズ。すべての画像をこの大きさにリサイズ
	# batch_size：一度に処理する画像の枚数。
	# class_mode：返すラベルの配列の型。binaryは2値分類のための値
	# shuffle：入力データをシャッフルするかどうか
    generator = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=32,
            shuffle=False,
            class_mode='categorical')

    # データの特徴量を算出し保存
    bottleneck_features = model.predict_generator(generator, NB_SAMPLES)
    np.save(open('h5py/bottleneck_features_test_3.npy', 'wb'), bottleneck_features)
    
    LOGGER.debug('save features.')

def train_top_model():
    input_data = np.load(open('h5py/bottleneck_features_test_3.npy', 'rb'))
    input_labels = np.array([0] * int((NB_SAMPLES / 2)) + [1] * int((NB_SAMPLES / 2)))
    input_labels = np_utils.to_categorical(input_labels, 15)
    print(input_labels)

    # 2値分類に有用な層を設定
    model = Sequential()
    model.add(Flatten(input_shape=input_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    
    model.summary()

    # modelのコンパイル
    # 第1引数：最適化手法＝RMSProp、第2引数：損失関数、第3引数：評価指標＝正解率
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # コンパイルしたModelに学習させる
    model.fit(input_data, input_labels,
              nb_epoch=NB_EPOCH, batch_size=32,
              validation_split=0.2)
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)

    LOGGER.debug('fit model.')
    
    # 学習済のモデルをJSON/YAML形式で保存
    LOGGER.debug('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join('model/','cnn_model_test_3.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join('model/','cnn_model_test_3.yaml'), 'w').write(yaml_string)

    LOGGER.debug('save learned model.')

save_bottlebeck_features()
train_top_model()
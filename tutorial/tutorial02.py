#!/usr/bin/env python35
# -*- coding: utf-8 -*-

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.utils.np_utils import convert_kernel
import tensorflow as tf


# 読み込むモデルの重みファイル、書き込むモデルの重みファイルのパスを指定
WEIGHTS_PATH = '../data/vgg16_weights.h5'
TOP_MODEL_WEIGHTS_PATH = 'bottleneck_fc_model.h5'

# 画像サイズを指定
IMG_WIDTH, IMG_HEIGHT = 150, 150

TRAIN_DATA_DIR = '../data/train'
VALIDATION_DATA_DIR = '../data/validation'
NB_TRAIN_SAMPLES = 2000
NB_VALIDATION_SAMPLES = 800
# 学習回数を指定
NB_EPOCH = 50


def save_bottleneck_features():
    # 画像の拡大縮小の値を変数に格納
    datagen = ImageDataGenerator(rescale=1./255)

    # VGG16ネットワークモデルを構築
	# sequentialモデルを使用
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
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
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
    print('Model loaded.')

    # 学習用データを読み込ませる
	# directory：ディレクトリへのパス。分類ごとのサブディレクトリを含み、
    #            そのサブディレクトリにPNGかJPG形式の画像が含まれていなければならない。
	# target_size：画像のサイズ。すべての画像をこの大きさにリサイズ
	# batch_size：一度に処理する画像の枚数。
	# class_mode：返すラベルの配列の型。binaryは2値分類のための値
	# shuffle：入力データをシャッフルするかどうか
    generator = datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=32,
            class_mode=None,
            shuffle=False)

    # 学習用データの特徴量を算出し保存
    bottleneck_features_train = model.predict_generator(generator, NB_TRAIN_SAMPLES)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    
    # 検証用データを読み込ませる
	# 各値は学習用データと同様
    generator = datagen.flow_from_directory(
            VALIDATION_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    
    # 検証用データの特徴量を算出し保存
    bottleneck_features_validation = model.predict_generator(generator, NB_VALIDATION_SAMPLES)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * int((NB_TRAIN_SAMPLES / 2)) + [1] * int((NB_TRAIN_SAMPLES / 2)))
    
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * int((NB_VALIDATION_SAMPLES / 2)) + [1] * int((NB_VALIDATION_SAMPLES / 2)))
    
    # 2値分類に有用な層を設定
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # modelのコンパイル
    # 第1引数：最適化手法＝RMSProp、第2引数：損失関数=binary_crossentropy、第3引数：評価指標＝正解率
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    # コンパイルしたModelに学習させる
    model.fit(train_data, train_labels,
              nb_epoch=NB_EPOCH, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)
    
    # 学習済のモデルをJSON/YAML形式で保存
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join('./','cnn_model.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join('./','cnn_model.yaml'), 'w').write(yaml_string)

save_bottleneck_features()
train_top_model()

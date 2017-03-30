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
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten , Dense

# 読み込むモデルの重みファイル、書き込むモデルの重みファイルのパスを指定
WEIGHTS_PATH = '../data/vgg16_weights.h5'
TOP_MODEL_WEIGHTS_PATH = 'bottleneck_fc_model.h5'

# 画像サイズを指定
IMG_WIDTH, IMG_HEIGHT = 150,150


TRAIN_DATA_DIR = '../data/train'
VALIDATION_DATA_DIR = '../data/validation'
NB_TRAIN_SAMPLES = 2000
NB_VALIDATION_SAMPLES = 800
NB_EPOCH = 50

# VGG16ネットワークモデルを構築
# sequentialモデルを使用
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

# 2値分類に有用なモデルを設定
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# モデルの重みを読み込み
# モデルの微調整はこのファイルのみ上書きする
top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

# モデルをセット
model.add(top_model)

# 第25層までのモデルについて、学習後に重みの上書きがされないようにする
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
# modelのコンパイル
# 第1引数：損失関数、 第2引数：最適化手法、 第3引数：評価指標＝正解率
# 最適化手法の「lr」が学習率。微調整のためにかなり小さい値になっている。
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# 学習用データの加工値を設定
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# 検証用データの加工値を設定
test_datagen = ImageDataGenerator(rescale=1./255)

# 学習用データを読み込ませる
# directory：ディレクトリへのパス。分類ごとのサブディレクトリを含み、
#            そのサブディレクトリにPNGかJPG形式の画像が含まれていなければならない。
# target_size：画像のサイズ。すべての画像をこの大きさにリサイズ
# batch_size：一度に処理する画像の枚数
# class_mode：返すラベルの配列の型。binaryは2値分類のための値
train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary')

# 検証用データを読み込ませる
# 各値は学習用データと同様
validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary')

# コンパイルしたModelに学習させ、モデルの微調整をする
model.fit_generator(
        train_generator,
        samples_per_epoch=NB_TRAIN_SAMPLES,
        nb_epoch=NB_EPOCH,
        validation_data=validation_generator,
        nb_val_samples=NB_VALIDATION_SAMPLES)

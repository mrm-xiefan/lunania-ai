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

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# 画像サイズを指定
IMG_WIDTH, IMG_HEIGHT = 150, 150


TRAIN_DATA_DIR = './../data/train'
VALIDATION_DATA_DIR = './../data/validation'
NB_TRAIN_SAMPLES = 2000
NB_VALIDATION_SAMPLES = 800
# 学習回数を指定
NB_EPOCH = 2

# sequentialモデルを構築
# 最初の層のみ入力データの形式の記述が必要
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2値分類に有用な層を設定
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# modelのコンパイル
# 第1引数：損失関数＝?、第2引数：最適化手法=RMSProp、第3引数：評価指標＝正解率
model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

# 学習用データの加工
# rescale：拡大縮小、shear_range：せん断(1辺を固定し、対辺の平行を保ったまま変形)、
# zoom_range：ズーム、horizontal_flip=True；水平方向にランダムに平行移動)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# 検証用データの加工
test_datagen = ImageDataGenerator(rescale=1. / 255)


# 学習用データを読み込ませる
# directory：ディレクトリへのパス。分類ごとのサブディレクトリを含み、
#            そのサブディレクトリにPNGかJPG形式の画像が含まれていなければならない。
# target_size：画像のサイズ。すべての画像をこの大きさにリサイズ
# batch_size：一度に処理する画像の枚数
# class_mode：返すラベルの配列の型。binaryは2値分類のための値
#train_generator = train_datagen.flow_from_directory(
#    TRAIN_DATA_DIR,
#    target_size=(IMG_WIDTH, IMG_HEIGHT),
#    batch_size=16,
#    class_mode='binary')


# 検証用データを読み込ませる
# 各値は学習用データと同様
#validation_generator = test_datagen.flow_from_directory(
#    VALIDATION_DATA_DIR,
#    target_size=(IMG_WIDTH, IMG_HEIGHT),
#    batch_size=16,
#    class_mode='binary')

i = 0;
for batch in train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=2,
    class_mode='binary',
    save_to_dir='pre',
    save_prefix='s'):
    i += 1
    if i > 1:
        break


# コンパイルしたModelに学習させる
#model.fit_generator(
#    train_generator,
#    samples_per_epoch=NB_TRAIN_SAMPLES,
#    nb_epoch=NB_EPOCH,
#    validation_data=validation_generator,
#    nb_val_samples=NB_VALIDATION_SAMPLES)


# 引数に指定されたパスに、HDF5形式のモデルの重みファイルを保存
#model.save_weights('first_try.h5')

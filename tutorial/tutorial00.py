#!/usr/bin/env python35
# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 入力データの加工値を設定
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


img = load_img('./cat.1.jpg')

# 画像データをNumpy配列にする(3, 150, 150)
x = img_to_array(img)
# Numpy配列をリサイズ(1, 3, 150, 150)
x = x.reshape((1,) + x.shape)


# 加工した画像をpreviewディレクトリに保存
# 1枚の画像から21枚の加工データを作成
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break

#!/usr/bin/env python35
# -*- coding: utf-8 -*-
import os
import logging
import argparse
import numpy as np
import ast

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler())

# keras CPU only  mode.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    return parser

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()

    from keras import backend as K
    from keras.preprocessing.image import load_img, img_to_array
    from keras.layers import Dense, Activation, Input
    from keras.layers.recurrent import LSTM
    from keras.models import Model, Sequential

    span = 100
    # create model
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(
        None, 1, span), return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mean_squared_error",
                  optimizer="adam", metrics=['accuracy'])

    data = []
    with open('./rnn.test.txt', 'r') as f:
        data = f.read()

    data_list = list(map(lambda x: float(x), data.split('\n')[:-1]))
    # normalization
    copy = np.copy(data_list)
    std_data_list = (copy - copy.mean()) / copy.std()

    train = []
    label = []
    for i in range(0, len(std_data_list) - (span + 1)):
        train.append([std_data_list[i:i + span]])
        label.append(std_data_list[i + span])

    LOGGER.debug(train[0])
    LOGGER.debug(label[0])

    train = np.asarray(train)
    label = np.asarray(label)
    LOGGER.debug(train.shape)
    LOGGER.debug(label.shape)

    model.summary()
    model.fit(train, label, batch_size=300,
              nb_epoch=10, validation_split=0.2)

    result = model.predict(train)
    pre = (result * copy.std()) + copy.mean()
    with open('./result.txt', 'w') as f:
        f.writelines(list(map(lambda x: str(x[0]) + '\r\n', pre)))

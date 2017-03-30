#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import time
import uuid

from keras.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class LearningTime():

    def __init__(self):
        self.learning_start = 0  # type: Float
        self.learning_end = 0  # type: Float


class LearningTimeLogger(Callback):
    '''
    学習時間を記録するためのLOGGERクラス
    fit時のコールバックとして利用する。
    '''

    def __init__(self, is_log_batch=True):
        self.learn = LearningTime()  # type: LearningTime
        self.epoch = []  # type: List[LearningTime]
        self.batch = []  # type: List[LearningTime]
        self.is_log_batch = is_log_batch  # type: Bool
        super().__init__()

    def on_train_begin(self, logs={}):
        self.learn.learning_start = time.time()

    def on_train_end(self, logs={}):
        self.learn.learning_end = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        learn = LearningTime()
        learn.learning_start = time.time()
        self.epoch.append(learn)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch[-1].learning_end = time.time()

    def on_batch_begin(self, batch, logs={}):
        if self.is_log_batch:
            learn = LearningTime()
            learn.learning_start = time.time()
            self.batch.append(learn)

    def on_batch_end(self, batch, logs={}):
        if self.is_log_batch:
            self.batch[-1].learning_end = time.time()


class RemoteNotificationLogger(Callback):
    '''
    リモートサーバに対して現在の学習状況を通知するためのLOGGERクラス
    '''

    def __init__(self, endpoint: str='http://localhost:5000',
                 api: str=''):
        super().__init__()
        try:
            import requests
            self._requests = requests
        except ImportError as e:
            LOGGER.error(self.__class__.__name__ +
                         ' is request module required.')
            raise e

        self._endpoint = endpoint
        self._api = api

    def on_train_begin(self, logs={}):
        self._requests.post(self._endpoint + self._api,
                            json={
                                'learningStartTime': time.time(),
                                'epoch': self.params['nb_epoch']
                            })

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

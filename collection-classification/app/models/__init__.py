#!/usr/bin/env python35
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from typing import Optional
from keras.models import Model

from app.core.error import UnsupportedOperationError


class ModelContainer():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_model(self, input_shape) -> Model:
        pass

    @abstractmethod
    def compile(self, model: Model) -> None:
        pass

    @abstractmethod
    def get_model_key(self) -> str:
        return self.__class__.__name__


class SavedModelContainer(ModelContainer):

    def __init__(self, model_key: str, model: Model, is_all_in_one: bool):
        self._model_key = model_key
        self._model = model
        self._is_all_in_one = is_all_in_one

    def get_model(self, input_shape=None) -> Model:
        return self._model

    def compile(self, model: Model) -> None:
        raise UnsupportedOperationError('Unsuppoerted function.')

    def get_model_key(self) -> str:
        return self._model_key

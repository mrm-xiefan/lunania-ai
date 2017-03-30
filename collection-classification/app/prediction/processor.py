#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from app.core.error import UnsupportedOperationError
from typing import Generic, List, TypeVar, Tuple

LOGGER = logging.getLogger(__name__)

E = TypeVar('E')
R = TypeVar('R')


class PredictionResultProcessor(Generic[E, R]):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def process(self, predicton_result: E) -> R:
        raise UnsupportedOperationError('no implementation')


class ClassificationProcessor(PredictionResultProcessor[List[float], str]):

    def __init__(self, label: List[str], limit: int=None):
        super().__init__()
        self._label = label
        self._limit = limit

    def _get_label_params(self, result: List[float]):
        tmp = {}
        for name, acc in zip(self._label, result):
            tmp[name] = acc
        tmp = sorted(tmp.items(),
                     key=lambda x: x[1], reverse=True)

        label_params = {"labels": []}
        for label_param in tmp:
            label_params["labels"].append(
                {"name": label_param[0], "acc": label_param[1]})

        return label_params

    def _get_predict_labels(self, label_params: dict):
        labels = label_params["labels"]
        if self._limit is not None:
            if self._limit <= 0:
                raise ValueError(
                    'limit must be positive value.')
            elif self._limit > len(labels):
                raise ValueError(
                    'limit must be less than or equal to the number of label.')
            else:
                return labels[:self._limit]
        else:
            return labels

    def process(self, result: List[List[float]]) -> List[str]:
        label_params = self._get_label_params(result[0])
        predict_labels = self._get_predict_labels(label_params)

        return label_params, predict_labels


class CrfSegmentationProcessor(PredictionResultProcessor[List[float], str]):

    def __init__(self, size: Tuple[int],
                 fcn_classes: int):
        self._size = size
        self._fcn_classes = fcn_classes

        self._d = dcrf.DenseCRF2D(self._size[1], self._size[
            0], self._fcn_classes)
        super().__init__()

    def process(self, result: List[List[float]],
                img: np.ndarray=None) -> List[str]:
        if img is None:
            raise 'img is required.'

        result = result[0]

        # get unary potentials (neg log probability)
        label = result.reshape(
            (self._size[1], self._size[0], self._fcn_classes)).transpose((2, 0, 1))
        U = unary_from_softmax(label)
        self._d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to
        # theCRF
        self._d.addPairwiseGaussian(sxy=(
            3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        self._d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13),
                                     rgbim=img,
                                     compat=10,
                                     kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = self._d.inference(5)
        # Find out the most probable class for each pixel.
        return np.argmax(Q, axis=0).reshape(self._size)

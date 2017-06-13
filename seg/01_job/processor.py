#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from luna import LunaExcepion
#from app.libs.ssd.ssd_utils import BBoxUtility
from typing import Generic, List, Tuple, TypeVar

LOGGER = logging.getLogger(__name__)

E = TypeVar('E')
R = TypeVar('R')


class PredictionResultProcessor(Generic[E, R]):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def process(self, predicton_result: E) -> R:
        raise LunaExcepion(config.unsupport)


class SegmentationProcessor(PredictionResultProcessor[List[float], str]):

    def __init__(self, size: Tuple[int]):
        '''
        argments
            size: (height, width)
        '''
        self._size = size
        super().__init__()

    def process(self, result: np.ndarray) -> np.ndarray:
        label_data = []

        for i in range(len(result[0])):
            label_data.append(np.argmax(result[0][i]))

        # reshape to original size
        return np.reshape(label_data, self._size)


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


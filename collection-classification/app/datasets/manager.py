#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import yaml
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)

from typing import Generator, Tuple

LOGGER = logging.getLogger(__name__)


class DatasetStore():
    '''
    学習用データLOADER
    '''

    def __init__(self, base_folder_path: str):
        self._base_folder_path = os.path.expanduser(base_folder_path)

    def load(self, dataset_key: str,
             size: Tuple[int],
             generator_option=None,
             flow_option=None) -> any:
        '''
        学習用データのload処理
        '''
        LOGGER.debug('load dataset:' + dataset_key)
        path = self._get_store_path(dataset_key)
        assert os.path.exists(path), 'Not exists folder. path:' + path

        # customize option
        if generator_option is None:
            generator_option = {'rescale': 1. / 255}

        if flow_option is None:
            flow_option = {}

        generator = ImageDataGenerator(**generator_option)
        generator = generator.flow_from_directory(path,
                                                  target_size=size,
                                                  **flow_option)

        return generator

    def saveFeature(self, dataset_key, feature, labels, class_count, class_indices):
        path = os.path.join(self._get_store_path(dataset_key), 'feature.npy')
        with open(path, 'wb') as feature_file:
            np.save(feature_file, feature)

        path = os.path.join(self._get_store_path(dataset_key), 'metadata.yaml')
        with open(path, 'w') as metadata_file:
            yaml.dump({'label': labels, 'classes': class_count,
                       'indices': class_indices}, stream=metadata_file)

    def loadFeature(self, dataset_key):
        path = os.path.join(self._get_store_path(dataset_key), 'feature.npy')
        with open(path, 'rb') as npy_file:
            feature = np.load(npy_file)

        # TODO load_feature_metadata に統合
        path = os.path.join(self._get_store_path(dataset_key), 'metadata.yaml')
        with open(path, 'r') as metadata_file:
            metadata = yaml.load(metadata_file)

        return [feature, metadata['label'], metadata['classes'], metadata['indices']]

    def load_feature_metadata(self, dataset_key: str):
        path = os.path.join(self._get_store_path(dataset_key), 'metadata.yaml')
        with open(path, 'r') as metadata_file:
            metadata = yaml.load(metadata_file)

        return metadata

    def _get_store_path(self, dataset_key: str):
        return os.path.join(self._base_folder_path, dataset_key)

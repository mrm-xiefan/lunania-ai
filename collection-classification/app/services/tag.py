#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import PIL as Image
from keras.models import Model
from keras.preprocessing.image import load_img

from app import context
from app.constants.color_palette import Label16Types, LabelEnum
from app.models import SavedModelContainer
from app.models.manager import ModelStore
from app.prediction.processor import (ClassificationProcessor,
                                      CrfSegmentationProcessor)
from app.utils.image_utils import LabelExtentProcessor, to_array_and_reshape
from typing import Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


class TagClassificationService():

    def __init__(self, model_store: ModelStore):

        self._model_store = model_store  # type: ModelStore

        self.input_image_size = context.get('tag.fcn.size.eval', tuple)
        self.fcn_class_count = context.get('tag.fcn.classes', int)
        self.item_labels = context.get('tag.fcn.labels', list)

        # fcn model
        LOGGER.info('load fcn...')
        self._fcn_model_container = self._model_store.load(
            context.get('tag.fcn.modelKey', str), use_all_in_one=False)
        self._fcn_predict_processor = CrfSegmentationProcessor(
            self.input_image_size, self.fcn_class_count)

        LOGGER.info('load dress length...')
        # dress length model
        self._dress_length_contaier = self._model_store.load(
            context.get('tag.dressLength.modelKey', str),
            use_all_in_one=False)
        self._dress_length_except = context.get(
            'tag.dressLength.except', list)
        self._dress_length_processor = ClassificationProcessor(
            context.get('tag.dressLength.labels', list))

        # sleeve length model
        self._sleeve_length_contaier = self._model_store.load(
            context.get('tag.sleeveLength.modelKey', str),
            use_all_in_one=False)
        self._sleeve_length_except = context.get(
            'tag.sleeveLength.except', list)
        self._sleeve_length_processor = ClassificationProcessor(
            context.get('tag.sleeveLength.labels', list))

        # bottom length model
        self._bottom_length_contaier = self._model_store.load(
            context.get('tag.bottomLength.modelKey', str),
            use_all_in_one=False)
        self._bottom_length_inculde = context.get(
            'tag.bottomLength.include', list)
        self._bottom_length_processor = ClassificationProcessor(
            context.get('tag.bottomLength.labels', list))

        LOGGER.info('load model flow...')
        # croped image classification models
        # type: Dist[str, Tuple[SavedModelContainer, ClassificationProcessor]]
        self._model_flow_def_on_croped_image = {}
        self._model_flow_index = [
            'color', 'material', 'detail', 'style', 'pattern']
        for clazz_type in self._model_flow_index:
            LOGGER.info('load model flow class:%s..', clazz_type)
            self._model_flow_def_on_croped_image[clazz_type] = (self._model_store.load(
                context.get('tag.{}.modelKey'.format(clazz_type), str), use_all_in_one=False),
                ClassificationProcessor(context.get('tag.{}.labels'.format(clazz_type), list)))

    def get_tags(self, image: Image.Image,
                 output_detection=None,
                 output_file_name='default') -> List['ImageTag']:

        image = image.resize(reversed(self.input_image_size))
        org_image = image
        image = to_array_and_reshape(image, samplewise=True)

        # fcnによる領域抽出処理
        fcn_model = self._fcn_model_container.get_model()
        fcn_result = fcn_model.predict_on_batch(image)

        _org = np.copy(np.asarray(org_image))
        LOGGER.debug('_org shape:%s', _org.shape)
        seg_label = self._fcn_predict_processor.process(fcn_result, _org)

        # 抽出したlabelデータを領域に変換する。
        label = LabelEnum.of(self.fcn_class_count)  # type: Label16Type
        lxp = LabelExtentProcessor(seg_label,
                                   label.get_colors(),
                                   (30, 30), (3, 2))

        if output_detection:
            label.to_image(seg_label).save(
                os.path.join(output_detection, '{}-crf.jpg'.format(output_file_name)))
            lxp.mark_all(org_image).save(
                os.path.join(output_detection, '{}-mark.jpg'.format(output_file_name)))

        LOGGER.debug('exists label:%s', lxp.get_exists_labels())
        tags = []  # type: List[ImageTag]
        for index in lxp.get_exists_labels():
            if index == '0':
                LOGGER.debug('skip background.')
                continue

            tag = ImageTag()
            tag.item = self.item_labels[int(index)]

            LOGGER.debug('predict item index:%s', index)
            tag.from_dict(self._predict_item_from_croped_image(
                lxp.crop(index, org_image)))

            tags.append(tag)

        # dress length predict.
        dress_tag = list(filter(
            lambda x: x.item not in self._dress_length_except, tags))
        if len(dress_tag) != 0:
            result = self._dress_length_contaier.get_model().predict(image)
            dress_tag[0].dress_length = self._dress_length_processor(result)

        # sleeve length predict.
        sleeve_tag = list(filter(
            lambda x: x not in self._sleeve_length_except, tags))
        if len(sleeve_tag) != 0:
            result = self._sleeve_length_contaier.get_model().predict(image)
            sleeve_tag[0].sleeve_length = self._sleeve_length_processor(result)

        # bottom length predict.
        bottom_tag = list(
            filter(lambda x: x in self._bottom_length_except, tags))
        if len(bottom_tag) != 0:
            result = self._bottom_length_contaier.get_model().predict(image)
            bottom_tag[0].bottom_length = self._bottom_length_processor(result)

        # end.
        return tags

    def _predict_item_from_croped_image(self, image: Image.Image) -> Dict[str, any]:
        '''
        カラー、素材、詳細、スタイル、柄のタグ判定処理
        '''
        stack = {}
        for key, v in self._model_flow_def_on_croped_image.items():
            model_container, processor = v

            LOGGER.debug('pipeline process. model_key:%s',
                         model_container._model_key)

            model = model_container.get_model()

            input_shape = model.input_shape
            _image = to_array_and_reshape(
                image.resize((input_shape[2], input_shape[1])))

            result = model.predict(_image)
            stack[key] = processor.process(result)

        return stack


class ImageMetadata():

    def __init__(self):

        self.brand = ''  # type: str
        self.collection = ''  # type: str
        self.path = ''  # type: str
        self.tags = []  # type: 'ImageTag'


class ImageTag():

    def __init__(self):

        self.item = ''  # type: str
        self.colors = []  # type: List[str]
        self.pattern = []  # type: List[str]
        self.dress_length = []  # type: List[str]
        self.sleeve_length = []  # type: List[str]
        self.bottom_length = []  # type: List[str]
        self.details = []  # type: List[str]
        self.materials = []  # type: List[str]
        self.style = ''  # type: str

    def from_dict(self, result: Dict):

        self.colors = result['color']
        self.pattern = result['pattern']
        self.details = result['detail']
        self.materials = result['material']
        self.style = result['style']

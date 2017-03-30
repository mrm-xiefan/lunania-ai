#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import glob
import logging
import math
import os
import random

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image

LOGGER = logging.getLogger(__name__)


class LabelProcessor:

    def __init__(self, seg_img, label_colors: list):
        self.seg_img = seg_img
        self.label_colors = label_colors


class LabelExtentProcessor(LabelProcessor):

    def __init__(self, seg_img, label_colors: list, margin_size: tuple =None, aspect_ratio: tuple = None):
        '''
        margin_size: トリミング時のマージンの大きさ。[0]:上下のマージン、[1]:左右のマージン
        aspect_ratio: トリミング時のアスペクト比。[0]:高さ、[1]:幅
        '''
        LabelProcessor.__init__(self, seg_img, label_colors)
        self.margin_size = margin_size
        self.aspect_ratio = aspect_ratio

        self.label_extent_map = self._get_label_extent_map(
            self.seg_img, self.label_colors, self.margin_size, self.aspect_ratio)

    def crop(self, label_index: int, image) -> Image.Image:
        '''
        引数に指定された画像について、分類ラベル単位で画像をトリミングする。
        '''
        _org_img = image

        if self.is_exists_label(label_index):
            top, bottom, left, right = self.label_extent_map[label_index]

            crop_image = _org_img.crop((left, top, right, bottom))

            return crop_image
        else:
            return None

    def crop_npy_data(self, label_index: int, org_npy: np.ndarray) -> np.ndarray:
        '''
        分類ラベル単位で画像をトリミングした際、
        ピクセル毎の値をNumpy配列にて返却する。
        '''
        extent = self.label_extent_map[label_index]

        # アスペクト比を固定してトリミングした際に、元画像からはみ出す領域を取得
        protrusion = self._get_protrusion(extent, org_npy.shape)
        t_pro, b_pro, l_pro, r_pro = protrusion

        top, bottom, left, right = extent
        extent_h = bottom - top
        extent_w = right - left
        crop_npy = np.zeros((extent_h, extent_w, 1))

        # 元画像からはみ出した領域は、ピクセルの値を0のままにする
        for h in range(extent_h - t_pro - b_pro):
            for w in range(extent_w - l_pro - r_pro):
                index_h = t_pro + h
                index_w = l_pro + w
                crop_npy[index_h, index_w] = org_npy[
                    top + index_h, left + index_w]
        return crop_npy

    def _get_label_extent_map(self, seg_img, label_colors: list,
                              margin_size: tuple, aspect_ratio: tuple) -> dict:
        '''
        分割画像より各ラベルの領域を算出する。結果を辞書型にて返却する。
        return {label_index: (top, bottom, left, right)}
        '''
        seg_img_array = np.array(seg_img)
        seg_img_array = seg_img_array.astype(np.int64)
        image_height, image_width = seg_img_array.shape[:2]

        # (分類ラベル数, 4)のNumpy配列を生成
        # [x][0]:存在領域の上端のピクセル番号, [x][1]:存在領域の下端のピクセル番号
        # [x][2]:存在領域の左端のピクセル番号, [x][3]:存在領域の右端のピクセル番号
        # (xは任意の分類ラベル番号とする)
        label_extent_list = np.zeros((len(label_colors), 4))

        # (分類ラベル数, 4)のNumpy配列を生成
        # [x][0]:存在領域の上端のピクセル番号セット済フラグ, [x][1]:存在領域の下端のピクセル番号セット済フラグ
        # [x][2]:存在領域の左端のピクセル番号セット済フラグ, [x][3]:存在領域の右端のピクセル番号セット済フラグ
        # (xは任意の分類ラベル番号とする)
        # ピクセルの値をセットしたら、label_extent_listと対応するインデックスの値を1(True)にする。
        set_pixel_flg_list = np.zeros((len(label_colors), 4))

        for h in range(image_height):
            for w in range(image_width):
                label_index = seg_img_array[h, w]
                flg = set_pixel_flg_list[label_index, 0]
                if not flg:
                    label_extent_list[label_index, 0] = h
                    set_pixel_flg_list[label_index, 0] = 1

                label_index = seg_img_array[
                    image_height - 1 - h, image_width - 1 - w]
                flg = set_pixel_flg_list[label_index, 1]
                if not flg:
                    label_extent_list[label_index, 1] = image_height - 1 - h
                    set_pixel_flg_list[label_index, 1] = 1

        for w in range(image_width):
            for h in range(image_height):
                label_index = seg_img_array[h, w]
                flg = set_pixel_flg_list[label_index, 2]
                if not flg:
                    label_extent_list[label_index, 2] = w
                    set_pixel_flg_list[label_index, 2] = 1

                label_index = seg_img_array[
                    image_height - 1 - h, image_width - 1 - w]
                flg = set_pixel_flg_list[label_index, 3]
                if not flg:
                    label_extent_list[label_index, 3] = image_width - 1 - w
                    set_pixel_flg_list[label_index, 3] = 1

        # 画像内に存在するラベルの領域のみ、辞書型の変数に詰め込む。
        label_extent_dict = {}
        for i, extent in enumerate(label_extent_list):
            if set_pixel_flg_list[i][0] and set_pixel_flg_list[i][2]:
                # マージンが指定されている場合
                if margin_size:
                    extent = self._set_margin(
                        extent, margin_size, image_height, image_width)
                # アスペクト比が指定されている場合
                if aspect_ratio:
                    extent = self._fit_aspect(extent, aspect_ratio, i)

                label_extent_dict[str(i)] = tuple(map(int, extent))

        return label_extent_dict

    def _set_margin(self, label_extent: tuple, margin_size: tuple, image_height: int, image_width: int) -> tuple:
        '''
        指定された領域抽出のマージン部分を設定する。
        マージンを設定した結果をタプル型にて返却する。
        '''
        top, bottom, left, right = label_extent

        height_margin, width_margin = margin_size

        if top - height_margin >= 0 and bottom + height_margin < image_height:
            top = top - height_margin
            bottom = bottom + height_margin

        if left - width_margin >= 0 and right + width_margin < image_width:
            left = left - width_margin
            right = right + width_margin

        return top, bottom, left, right

    def _fit_aspect(self, label_extent: tuple, aspect_ratio: tuple, index: int) -> tuple:
        '''
        分類ラベルを含み、指定されたアスペクト比でトリミングした際の領域を算出する。
        算出した上下左右の領域をタプル型にて返却する。
        '''
        top, bottom, left, right = label_extent

        h = bottom - top + 1
        w = right - left + 1
        extent_ratio = tuple(map(int, (h, w)))

        aspect_h, aspect_w = aspect_ratio

        # トリミング時に、抽出領域のheightを拡大するかを判定
        isExpandingHeight = self._is_expanding_h(
            aspect_ratio, extent_ratio)

        if isExpandingHeight:
            expand_h = w * aspect_h / aspect_w - h
            expand_w = 0
            # heightの拡大領域を2で割った際、割り切れなかったら切り上げる
            expand_top_len = math.ceil(expand_h / 2)
            # heightの拡大領域からtopの拡大領域を引き、bottomの拡大領域を算出
            expand_bottom_len = expand_h - expand_top_len
            expand_left_len = 0
            expand_right_len = 0
        else:
            expand_h = 0
            expand_w = h * aspect_w / aspect_h - w
            expand_top_len = 0
            expand_bottom_len = 0
            # leftの拡大領域を2で割った際、割り切れなかったら切り上げる
            expand_left_len = math.ceil(expand_w / 2)
            # widthの拡大領域からleftの拡大領域を引き、rightの拡大領域を算出
            expand_right_len = expand_w - expand_left_len

        expand_top = top - expand_top_len
        expand_bottom = bottom + expand_bottom_len
        expand_left = left - expand_left_len
        expand_right = right + expand_right_len

        return expand_top, expand_bottom, expand_left, expand_right

    def _is_expanding_h(self, aspect_ratio: tuple, extent_ratio: tuple) -> bool:
        '''
        設定されたアスペクト比でトリミングする際に、
        heightを余分に取ってトリミングするかどうかをboolにて返却する。
        aspect_ratio: ユーザにより設定されたアスペクト比
        extent_ratio: 領域抽出範囲の縦横比(margin_sizeが設定されている場合、マージンを加えている)
        '''
        aspect_h, aspect_w = aspect_ratio
        extent_h, extent_w = extent_ratio

        # aspect_w_to_h: aspect_ratioにおける、縦に対する横の比
        # extent_w_to_h: extent_ratioにおける、縦に対する横の比
        aspect_w_to_h = aspect_w / aspect_h
        extent_w_to_h = extent_w / extent_h

        # aspect_w_to_h、extent_w_to_hの大小関係で、トリミング時のheightを大きくするか決める。
        # 例 aspect_w_to_h:0.5、 extent_w_to_h:0.7 -> expect_hを大きくして比の値を等しくする。
        # 例 aspect_w_to_h:0.7、 extent_w_to_h:0.5 -> expect_wを大きくして比の値を等しくする。
        if aspect_w_to_h <= extent_w_to_h:
            return True
        if aspect_w_to_h > extent_w_to_h:
            return False

    def _get_protrusion(self, extent: tuple, org_npy_shape: tuple) -> tuple:
        '''
        設定されたアスペクト比でトリミングする際に、
        元画像の大きさからはみ出す領域を算出する。
        領域をタプル型にて返却する。
        extent: アスペクト比を固定してトリミングする際の抽出領域
                元画像の大きさからはみ出ていると、top、leftは負の値、
                bottom、rightはそれぞれ元画像の高さ、幅の大きさ以上の値になる。
        org_npy_shape: 元画像の大きさ([0]: 高さ、[1]: 幅)
        '''
        top, bottom, left, right = extent
        org_height = org_npy_shape[0]
        org_width = org_npy_shape[1]

        # 元画像の大きさからはみ出す領域を算出。
        # はみ出ていなければ、値は0になる。
        # 例 top:    元画像の大きさからはみ出ていると、負の値になるため、
        #            0 - topにより、はみ出す領域を算出できる。
        # 例 bottom: 元画像の大きさからはみ出ていると、元画像の高さ以上の値になるため、
        #            bottom + 1 - org_heightにより、はみ出す領域を算出できる。
        t_pro = 0 - top if top < 0 else 0
        b_pro = bottom + 1 - org_height if bottom > org_height else 0
        l_pro = 0 - left if left < 0 else 0
        r_pro = right + 1 - org_width if right > org_width else 0

        return t_pro, b_pro, l_pro, r_pro

    def mark(self, label_index: int, image):

        _org_img = np.array(image)
        if not self.is_exists_label(label_index):
            raise
        else:
            color = self.label_colors[label_index]
            top, bottom, left, right = self.label_extent_map[label_index]

            # 矩形範囲の上下の辺を画像に追加
            _org_img[top,    left:right + 1] = color
            _org_img[bottom, left:right + 1] = color

            # 矩形範囲の左右の辺を画像に追加
            _org_img[top: bottom + 1, left] = color
            _org_img[top: bottom + 1, right] = color

            return Image.fromarray(np.uint8(_org_img))

    def mark_all(self, image):
        _image = np.array(image)
        for label in self.get_exists_labels():
            _image = self._mark(label, _image)

        return _image

    def is_exists_label(self, label_index: int) -> bool:
        return label_index in self.label_extent_map

    def get_exists_labels(self) -> set:
        return self.label_extent_map.keys()


class LabelFillingProcessor(LabelProcessor):

    def __init__(self, filling_color: np.ndarray =[0, 0, 0]):
        self.filling_color = filling_color

    def fill(self, image, seg_image: np.ndarray):
        '''
        画像と、その画像に対応する分類ラベルのインデックスの値を持つNumpy配列から、
        画像内でbackgroundに分類されている部分を塗りつぶす。
        塗りつぶす色は、LabelFillingProcessorのインスタンス生成時に設定する。
        (デフォルトは、black:[0, 0, 0])
        '''
        image = np.asarray(image)
        filling_image = np.zeros(image.shape)
        for h, tmp in enumerate(filling_image):
            for w, pixel_color in enumerate(tmp):
                if int(seg_image[h, w]) == 0:
                    filling_image[h, w] = self.filling_color
                else:
                    filling_image[h, w] = image[h, w]

        return Image.fromarray(np.uint8(filling_image))


def to_array_and_reshape(image: Image.Image, rescale=True, samplewise=False) -> np.ndarray:
    image = img_to_array(image)  # type: np.ndarray
    if samplewise:
        image -= np.mean(image, axis=2, keepdims=True)

    if rescale:
        image /= 255

    return image.reshape((1,) + image.shape)


def increase_img_data(org_dir: str, save_dir: str, sample_num: int,
                      generator_option=None,
                      flow_option=None):

    classes = []
    for subdir in sorted(os.listdir(org_dir)):
        if os.path.isdir(os.path.join(org_dir, subdir)):
            classes.append(subdir)

    # customize option
    if generator_option is None:
        generator_option = {}

    if flow_option is None:
        flow_option = {}

    generator = ImageDataGenerator(**generator_option)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for clazz in classes:
        LOGGER.info('increase class:%s', clazz)
        save_sub_dir = os.path.join(save_dir, clazz)
        if not os.path.exists(save_sub_dir):
            os.mkdir(save_sub_dir)

        flow = generator.flow_from_directory(org_dir,
                                             classes=[clazz],
                                             batch_size=1,
                                             save_to_dir=save_sub_dir,
                                             **flow_option)
        for i in range(0, sample_num):
            flow.next()


def divide_to_validation(org_dir: str, divide_dir: str, divide_num: int):
    if not os.path.exists(divide_dir):
        os.mkdir(divide_dir)

    dirs = list(map(os.path.abspath, glob.glob(
        os.path.join(org_dir, '*'))))

    for dir in dirs:
        label = os.path.basename(dir)
        LOGGER.info('target label: %s', label)

        save_dir = os.path.join(divide_dir, label)
        if not os.path.exists(save_dir):
            LOGGER.debug(save_dir)
            os.mkdir(save_dir)

        files = list(map(os.path.abspath, glob.glob(
            os.path.join(dir, '*'))))
        random.shuffle(files)
        for i in range(divide_num):
            img = Image.open(files[i])
            base_name = os.path.basename(files[i])
            img.save(os.path.join(save_dir, base_name))
            os.remove(files[i])


def copy_from_crop_dir(crop_img_dir: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tops_dir = os.path.join(crop_img_dir, '1', 'tops')
    bottoms_dir = os.path.join(crop_img_dir, '2', 'bottoms')

    tags = os.listdir(tops_dir)
    for tag in tags:
        tag_img_count = 0
        tag_dir = os.path.join(save_dir, tag)
        if not os.path.exists(tag_dir):
            os.mkdir(tag_dir)

        # copy tops image from crop dir
        tops_imgs = list(map(os.path.abspath, glob.glob(
            os.path.join(tops_dir, tag, '*.jpg'))))
        for tops_img in tops_imgs:
            LOGGER.info('tops image path: %s', tops_img)
            img_name, ext = os.path.splitext(os.path.basename(tops_img))
            img = Image.open(tops_img)
            img.save(os.path.join(tag_dir, img_name + 't' + ext))
            tag_img_count += 1

        # copy bottoms image from crop dir
        bottoms_imgs = list(map(os.path.abspath, glob.glob(
            os.path.join(bottoms_dir, tag, '*.jpg'))))
        for bottoms_img in bottoms_imgs:
            LOGGER.info('bottoms image path: %s', bottoms_img)
            img_name, ext = os.path.splitext(os.path.basename(bottoms_img))
            img = Image.open(bottoms_img)
            img.save(os.path.join(tag_dir, img_name + 'b' + ext))
            tag_img_count += 1

        LOGGER.info('%s tag image num: %d', tag, tag_img_count)

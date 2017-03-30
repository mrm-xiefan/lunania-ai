#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os
from itertools import product
from random import gauss

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFilter


LOGGER = logging.getLogger(__name__)


class PreprocessService:

    def __init__(self):
        pass

    def inside_out(self, org_img: Image.Image, seg_img: np.array):
        '''
        引数で渡された画像、アノテーション画像を左右反転する。
        '''
        org_img = np.asarray(org_img)
        org_img_shape = org_img.shape
        seg_img_shape = seg_img.shape

        inside_out_org_img = np.zeros(org_img_shape)
        inside_out_seg_img = np.zeros(seg_img_shape)

        height = org_img_shape[0]
        width = org_img_shape[1]

        # inside out image
        for h in range(height):
            for w in range(width):
                inside_out_org_img[h, w] = org_img[
                    h, width - 1 - w]
                inside_out_seg_img[h, w] = seg_img[
                    h, width - 1 - w]
        return inside_out_org_img, inside_out_seg_img

    def upside_down(self, org_img: Image.Image, seg_img: np.array):
        '''
        引数で渡された画像、アノテーション画像を上下反転する。
        '''
        org_img = np.asarray(org_img)
        org_img_shape = org_img.shape
        seg_img_shape = seg_img.shape

        upside_down_org_img = np.zeros(org_img_shape)
        upside_down_seg_img = np.zeros(seg_img_shape)

        height = org_img_shape[0]
        width = org_img_shape[1]

        # upside down image
        for h in range(height):
            for w in range(width):
                upside_down_org_img[h, w] = org_img[
                    height - 1 - h, w]
                upside_down_seg_img[h, w] = seg_img[
                    height - 1 - h, w]
        return upside_down_org_img, upside_down_seg_img

    def zoom(self, org_img: Image.Image, seg_img: np.array, zoom_matrix: np.array):
        '''
        ズームイン、ズームアウト処理
        '''
        height, width = seg_img.shape[0], seg_img.shape[1]

        org_img = np.asarray(org_img)
        zoom_org_img = self._zoom(org_img, zoom_matrix, height, width)
        zoom_org_img = Image.fromarray(np.uint8(zoom_org_img))

        # segmentation image reshape to (height, width, channel)
        # from (height, width) for self._zoom()
        seg_img = np.reshape(seg_img, (height, width, 1))
        zoom_seg_img = self._zoom(
            seg_img, zoom_matrix, height, width)
        # segmentation image reshape to (height, width)
        # from (height, width, channel) for saving
        zoom_seg_img = np.reshape(zoom_seg_img, (height, width))

        return zoom_org_img, zoom_seg_img

    def _zoom(self, img: np.array, zoom_matrix: np.array, height: int, width: int):
        '''
        引数で渡された画像をズームイン、ズームアウトする。

        img.shape:(height, width, channel)
        '''
        o_x = float(height) / 2 + 0.5
        o_y = float(width) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(
            np.dot(offset_matrix, zoom_matrix), reset_matrix)

        zoom_img = np.rollaxis(img, 2, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                             final_offset, order=0, mode='nearest', cval=0) for x_channel in zoom_img]
        zoom_img = np.stack(channel_images, axis=0)
        zoom_img = np.rollaxis(zoom_img, 0, 3)

        return zoom_img

    def blur(self, org_img: Image.Image):
        '''
        画像にフィルタをかけてぼやかす処理
        '''
        return org_img.filter(ImageFilter.BLUR)

    def add_noise(self, org_img: Image.Image, amplification: int=80):
        '''
        画像にノイズを付加する処理
        '''
        width = org_img.size[0]
        height = org_img.size[1]
        org_pix = org_img.load()

        add_noise_image = Image.new("RGB", org_img.size)
        add_noise_pix = add_noise_image.load()

        for x, y in product(*map(range, (width, height))):
            noised_colors = map(lambda x: gauss(
                x, amplification), org_pix[x, y])
            noised_colors = map(lambda x: max(0, x), map(
                lambda x: min(255, x), noised_colors))
            noised_colors = tuple(map(int, noised_colors))
            add_noise_pix[x, y] = noised_colors

        return add_noise_image

    def process_samplewise(self, org_img: Image.Image):
        '''
        各座標のピクセル値について、RGBの平均を元の座標値から差し引く処理。(samplewise)
        '''
        org_img = np.asarray(org_img)
        org_img_shape = org_img.shape

        samplewise_org_img = np.zeros(org_img_shape)

        height = org_img_shape[0]
        width = org_img_shape[1]

        for h in range(height):
            for w in range(width):
                rgb_sum = org_img[h, w, 0] + \
                    org_img[h, w, 1] + org_img[h, w, 2]
                rgb_avg = (rgb_sum) / 3 if rgb_sum else 0
                samplewise_org_img[h, w] = org_img[h, w] - rgb_avg

        return Image.fromarray(np.uint8(samplewise_org_img))

    def minus_all_pix_avg(self, org_img: Image.Image):
        '''
        各座標のピクセル値について、画像全体のピクセル値の平均を差し引く処理
        '''
        org_img = np.asarray(org_img)
        org_img_shape = org_img.shape

        samplewise_org_img = np.zeros(org_img_shape)

        height = org_img_shape[0]
        width = org_img_shape[1]

        r_sum = 0
        g_sum = 0
        b_sum = 0
        for h in range(height):
            for w in range(width):
                r_sum += org_img[h, w, 0]
                g_sum += org_img[h, w, 1]
                b_sum += org_img[h, w, 2]
        all_rgb_avg = (r_sum / 3, g_sum / 3, b_sum / 3)
        for h in range(height):
            for w in range(width):
                samplewise_org_img[h, w] = org_img[h, w] - all_rgb_avg

        return Image.fromarray(np.uint8(samplewise_org_img))

    def get_exists_labels(self, seg_img: np.array):
        '''
        対象画像に存在するラベルを、アノテーション画像のNumpyから取得する。
        '''
        img_label = []
        for h, tmp in enumerate(seg_img):
            for w, label in enumerate(tmp):
                if label not in img_label:
                    img_label.append(label)
        return img_label

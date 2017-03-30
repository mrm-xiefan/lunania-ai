#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import shutil

import numpy as np
from PIL import Image

# keras CPU only  mode.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

LOGGER = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)-8s] [%(name)s.%(funcName)s] - %(message)s'))
LOGGER.addHandler(sh)
LOGGER.setLevel(logging.INFO)


# FCN用の学習データを作成するためのスクリプト

def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(
        description='Transform to npy from fcn png.')
    parser.add_argument('imgDir', type=str, help="input image file dir.")
    parser.add_argument('segDir', type=str, help='input seg file dir')
    parser.add_argument('distDir', type=str, help='dit file dir')
    parser.add_argument('--debug', type=bool, help='set logging lebel debug.')

    return parser


def to_npy_from_png(file_path: str, dist_dir: str, dist_file_name: str):

    dist_file_path = os.path.join(dist_dir, dist_file_name + '.npy')
    if os.path.exists(dist_file_path):
        LOGGER.error('npy file already exists. skip this file:%s', file_path)
        return

    png = np.asarray(Image.open(file_path))  # type: np.ndarray
    seg = png[:, :, 0]
    np.save(dist_file_path, seg)


def get_dist_file_path(dist_dir: str, full_file_name: str):

    dist_file_path = os.path.join(dist_dir, full_file_name)
    file_name, ext = os.path.splitext(full_file_name)

    index = 0
    while os.path.exists(dist_file_path):
        dist_file_path = os.path.join(dist_dir,
                                      file_name + str(index) + '.' + ext)
        index += 1

    LOGGER.debug(dist_file_path)
    return dist_file_path


def main(args):
    SEG_EXTENSION = 'png'

    img_dir = args.imgDir
    seg_dir = args.segDir
    dist_dir = args.distDir

    LOGGER.info('main process start...')
    LOGGER.info('options img dir:%s', img_dir)
    LOGGER.info('options seg dir:%s', seg_dir)
    LOGGER.info('options dist dir:%s', dist_dir)

    LOGGER.info('walk start. root dir:%s', img_dir)

    file_count = 0
    success_file_count = 0
    for root, dirs, files in os.walk(img_dir):
        LOGGER.debug('walk progress... root:%s', root)
        common_path = root[len(img_dir) + 1:]
        LOGGER.debug('seg common path:%s', common_path)

        for full_file_name in files:
            file_count += 1

            file_name, ext = os.path.splitext(full_file_name)
            seg_file_path = os.path.join(seg_dir,
                                         common_path,
                                         file_name + '.' + SEG_EXTENSION)

            if not os.path.exists(seg_file_path):
                LOGGER.error(
                    'seg not exists. skip this file:%s', seg_file_path)
                continue

            dist_file_path = get_dist_file_path(dist_dir, full_file_name)
            dist_file_name, _ = os.path.splitext(
                os.path.basename(dist_file_path))
            # file move
            LOGGER.debug(os.path.join(root, full_file_name))
            shutil.copyfile(os.path.join(root, full_file_name),
                            dist_file_path)
            to_npy_from_png(seg_file_path, dist_dir, dist_file_name)

            success_file_count += 1
            LOGGER.info('move and transform complete:%s', dist_file_path)

    LOGGER.info('main process end... file_count:%s, success_count:%s',
                file_count, success_file_count)

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)
    main(args)

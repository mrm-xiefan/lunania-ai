#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import shutil
import glob

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


# FCN用の学習データのサイズを変換するためのスクリプト

def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(
        description='resize fcn jpg data and fcn npy data.')
    parser.add_argument('fcnDataDir', type=str, help="fcn data dir.")
    parser.add_argument('saveDir', type=str, help='save resize data dir')
    parser.add_argument('height', type=int, help='resize image height')
    parser.add_argument('width', type=int, help='resize image width')
    parser.add_argument('--debug', type=bool, help='set logging lebel debug.')

    return parser


def main(args):
    fcn_data_dir = args.fcnDataDir
    save_dir = args.saveDir
    height = args.height
    width = args.width

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)
    LOGGER.info('options resize height:%d', height)
    LOGGER.info('options resize width:%d', width)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)

    resize_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')

        if seg_img.shape != (height, width):
            seg_img = Image.fromarray(seg_img)

            resize_org_img = org_img.resize((width, height))
            resize_seg_img = seg_img.resize((width, height))

            resize_seg_img = np.asarray(resize_seg_img)

            resize_org_img.save(os.path.join(save_dir, base_name + ext))
            np.save(os.path.join(save_dir, base_name + '.npy'), resize_seg_img)

            resize_file_count += 1
        else:
            org_img.save(os.path.join(save_dir, base_name + ext))
            np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

    LOGGER.info('main process end... file_count:%s, resize_file_count:%s',
                file_count, resize_file_count)

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)
    main(args)

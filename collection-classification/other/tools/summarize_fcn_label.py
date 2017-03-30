#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import shutil

from pathlib import Path

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
        description='summarize label count.')
    parser.add_argument('segDir', type=str, help="input npy file dir.")
    parser.add_argument('--debug', type=bool, help='set logging lebel debug.')

    return parser


def main(args):

    seg_dir = args.segDir

    LOGGER.info('main process start...')
    LOGGER.info('options seg dir:%s', seg_dir)

    labels = ["背景", "オールインワン", "カットソー", "カーディガン", "コート", "シャツ", "ジャケット", "スカート",
              "ニット", "パンツ", "パーカ", "ビスチェ", "ブラウス", "ブルゾン", "プルオーバー", "ワンピース"]

    summary = [0 for i in range(len(labels))]
    dir_path = Path(seg_dir)
    for i, path in enumerate(dir_path.glob('*.npy')):
        npy = np.load(str(path))  # type: np.ndarray
        for index in range(len(labels)):
            where = np.where(npy == index)
            if (len(where[0]) != 0) and (len(where[1]) != 0):
                summary[index] += 1

        if (i % 50 == 0):
            LOGGER.info('completed :%s', i)

    LOGGER.info('main process end... summary:%s', {
                label: sum for label, sum in zip(labels, summary)})

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)
    main(args)

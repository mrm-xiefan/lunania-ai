#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import shutil
import glob

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from service.preprocess import PreprocessService

LOGGER = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)-8s] [%(name)s.%(funcName)s] - %(message)s'))
LOGGER.addHandler(sh)
LOGGER.setLevel(logging.INFO)

# FCN用の学習データを加工し、増加させるためのスクリプト


def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(
        description='process fcn data.')
    subparser = parser.add_subparsers()

    # inside_out parser
    inside_out_parser = subparser.add_parser(
        'inside_out', help='inside out fcn jpg data and fcn npy data.')
    inside_out_parser.add_argument(
        'fcn_data_dir', type=str, help="fcn data dir.")
    inside_out_parser.add_argument(
        'save_dir', type=str, help='save inside_out data dir')
    inside_out_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    inside_out_parser.set_defaults(is_debug=False)

    inside_out_parser.set_defaults(func=command_inside_out)

    # upside_down parser
    upside_down_parser = subparser.add_parser(
        'upside_down', help='upside down fcn jpg data and fcn npy data.')
    upside_down_parser.add_argument(
        'fcn_data_dir', type=str, help="fcn data dir.")
    upside_down_parser.add_argument(
        'save_dir', type=str, help='save upside_down data dir')
    upside_down_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    upside_down_parser.set_defaults(is_debug=False)

    upside_down_parser.set_defaults(func=command_upside_down)

    # zoom parser
    zoom_parser = subparser.add_parser(
        'zoom', help='zoom fcn jpg data and fcn npy data.')
    zoom_parser.add_argument('fcn_data_dir', type=str, help="fcn data dir.")
    zoom_parser.add_argument(
        'save_dir', type=str, help='save zoom data dir')
    zoom_parser.add_argument('magnification', type=float,
                             help='zoom magnification. magnification < 1.0 is zoom in. magnification > 1.0 is zoom out.')
    zoom_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    zoom_parser.set_defaults(is_debug=False)

    zoom_parser.set_defaults(func=command_zoom)

    # filter parser
    filter_parser = subparser.add_parser(
        'filter', help='blur fcn jpg data using filter.')
    filter_parser.add_argument('fcn_data_dir', type=str, help="fcn data dir.")
    filter_parser.add_argument(
        'save_dir', type=str, help='save blur data dir')
    filter_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    filter_parser.set_defaults(is_debug=False)

    filter_parser.set_defaults(func=command_filter)

    # noise parser
    noise_parser = subparser.add_parser(
        'noise', help='add noise to fcn jpg data')
    noise_parser.add_argument('fcn_data_dir', type=str, help="fcn data dir.")
    noise_parser.add_argument(
        'save_dir', type=str, help='save add noise data dir')
    noise_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    noise_parser.set_defaults(is_debug=False)

    noise_parser.set_defaults(func=command_noise)

    # samplewise parser
    samplewise_parser = subparser.add_parser(
        'samplewise', help='samplewise fcn jpg data')
    samplewise_parser.add_argument(
        'fcn_data_dir', type=str, help="fcn data dir.")
    samplewise_parser.add_argument(
        'save_dir', type=str, help='save samplewise data dir')
    samplewise_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    samplewise_parser.set_defaults(is_debug=False)

    samplewise_parser.set_defaults(func=command_samplewise)

    # minus_avg parser
    minus_avg_parser = subparser.add_parser(
        'minus_avg', help='minus all pixel average fcn jpg data')
    minus_avg_parser.add_argument(
        'fcn_data_dir', type=str, help="fcn data dir.")
    minus_avg_parser.add_argument(
        'save_dir', type=str, help='save minus all pixel average data dir')
    minus_avg_parser.add_argument(
        '--debug', dest='is_debug', action='store_true')
    minus_avg_parser.set_defaults(is_debug=False)

    minus_avg_parser.set_defaults(func=command_minus_avg)

    return parser


def command_inside_out(args):
    INSIDE_OUT_LABEL_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]

    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    inside_out_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(
            save_dir, base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        # acquire label index in segmentation image
        img_label = service.get_exists_labels(seg_img)

        for label in img_label:
            if label in INSIDE_OUT_LABEL_LIST:
                inside_out_org_img, inside_out_seg_img = service.inside_out(
                    org_img, seg_img)

                inside_out_org_img = Image.fromarray(
                    np.uint8(inside_out_org_img))
                inside_out_org_img.save(os.path.join(
                    save_dir, base_name + '-inside_out' + ext))
                np.save(os.path.join(save_dir, base_name +
                                     '-inside_out.npy'), inside_out_seg_img)

                inside_out_file_count += 1
                break

    LOGGER.info('main process end... file_count:%s, inside_out_file_count:%s',
                file_count, inside_out_file_count)


def command_upside_down(args):
    UPSIDE_DOWN_LABEL_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]

    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    upside_down_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(
            save_dir, base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        # acquire label index in segmentation image
        img_label = service.get_exists_labels(seg_img)

        for label in img_label:
            if label in UPSIDE_DOWN_LABEL_LIST:
                upside_down_org_img, upside_down_seg_img = service.upside_down(
                    org_img, seg_img)

                upside_down_org_img = Image.fromarray(
                    np.uint8(upside_down_org_img))
                upside_down_org_img.save(os.path.join(
                    save_dir, base_name + '-upside_down' + ext))
                np.save(os.path.join(save_dir, base_name +
                                     '-upside_down.npy'), upside_down_seg_img)

                upside_down_file_count += 1
                break

    LOGGER.info('main process end... file_count:%s, upside_down_file_count:%s',
                file_count, upside_down_file_count)


def command_zoom(args):
    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir
    magnification = args.magnification

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)
    LOGGER.info('options zoom magnification:%e', magnification)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    zoom_matrix = np.array([[magnification, 0, 0],
                            [0, magnification, 0],
                            [0, 0, 1]])

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    zoom_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(save_dir,  base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        zoom_org_img, zoom_seg_img = service.zoom(
            org_img, seg_img, zoom_matrix)

        zoom_org_img.save(os.path.join(
            save_dir,  base_name + '-zoom' + str(magnification) + ext))
        np.save(os.path.join(save_dir, base_name +
                             '-zoom' + str(magnification) + '.npy'), zoom_seg_img)

        zoom_file_count += 1

    LOGGER.info('main process end... file_count:%s, zoom_file_count:%s',
                file_count, zoom_file_count)


def command_filter(args):
    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    blur_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(save_dir,  base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        blur_org_img = service.blur(org_img)

        blur_org_img.save(os.path.join(
            save_dir,  base_name + '-blur' + ext))
        np.save(os.path.join(save_dir, base_name + '-blur.npy'), seg_img)

        blur_file_count += 1

    LOGGER.info('main process end... file_count:%s, blur_file_count:%s',
                file_count, blur_file_count)


def command_noise(args):
    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    noise_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(save_dir,  base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        add_noise_org_img = service.add_noise(org_img)

        add_noise_org_img.save(os.path.join(
            save_dir,  base_name + '-noise' + ext))
        np.save(os.path.join(save_dir, base_name + '-noise.npy'), seg_img)

        noise_file_count += 1

    LOGGER.info('main process end... file_count:%s, add_noise_file_count:%s',
                file_count, noise_file_count)


def command_samplewise(args):
    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    samplewise_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(save_dir,  base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        add_samplewise_org_img = service.process_samplewise(org_img)

        add_samplewise_org_img.save(os.path.join(
            save_dir,  base_name + '-samplewise' + ext))
        np.save(os.path.join(save_dir, base_name + '-samplewise.npy'), seg_img)

        samplewise_file_count += 1

    LOGGER.info('main process end... file_count:%s, samplewise_file_count:%s',
                file_count, samplewise_file_count)


def command_minus_avg(args):
    service = PreprocessService()

    fcn_data_dir = args.fcn_data_dir
    save_dir = args.save_dir

    LOGGER.info('main process start...')
    LOGGER.info('options fcn data dir:%s', fcn_data_dir)
    LOGGER.info('options save dir:%s', save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    files = list(map(os.path.abspath, glob.glob(
        os.path.join(fcn_data_dir, '*.jpg'))))
    file_count = len(files)
    minus_avg_file_count = 0

    for file in files:
        file_name, ext = os.path.splitext(file)
        base_name, ext = os.path.splitext(os.path.basename(file))

        org_img = Image.open(file)
        seg_img = np.load(file_name + '.npy')
        # save original image
        org_img.save(os.path.join(save_dir,  base_name + ext))
        np.save(os.path.join(save_dir, base_name + '.npy'), seg_img)

        add_minus_avg_org_img = service.minus_all_pix_avg(org_img)

        add_minus_avg_org_img.save(os.path.join(
            save_dir,  base_name + '-minus_avg' + ext))
        np.save(os.path.join(save_dir, base_name + '-minus_avg.npy'), seg_img)

        minus_avg_file_count += 1

    LOGGER.info('main process end... file_count:%s, minus_avg_file_count:%s',
                file_count, minus_avg_file_count)

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()
    if args.is_debug:
        LOGGER.setLevel(logging.DEBUG)

    # execute subcommand.
    args.func(args)

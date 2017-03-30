#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import shutil
from datetime import datetime as dt

from keras.backend import tensorflow_backend as tf
from PIL import Image

# app initialize
from app import context
from app.core.log import logging_time
from app.datasets.manager import DatasetStore
from app.models.bottom_length import (BottomLengthModelContainer,
                                      BottomLengthModelContainerAtResNet)
from app.models.manager import ModelStore
from app.prediction.processor import ClassificationProcessor
from app.services.bottom_length import BottomLengthClassificationService
from app.utils import image_utils
from typing import List

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

LOGGER = logging.getLogger(__name__)


def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(description='Process')
    subparser = parser.add_subparsers()

    # learn parser
    learn_parser = subparser.add_parser('learn', help='learn help')
    learn_parser.add_argument('train_dataset_key', type=str,
                              help='train dataset key')
    learn_parser.add_argument('validation_dataset_key', type=str,
                              help='validation dataset key')
    learn_parser.add_argument('class_num', type=int,
                              help='the number of classificaton class')
    learn_parser.add_argument('epoch', type=int, help='learning epoch')

    learn_parser.set_defaults(func=command_learn)

    # continue parser
    continue_parser = subparser.add_parser('continue', help='continue help')
    continue_parser.add_argument('model_key', type=str,
                                 help='model key to continue')
    continue_parser.add_argument('train_dataset_key', type=str,
                                 help='train dataset key')
    continue_parser.add_argument('validation_dataset_key', type=str,
                                 help='validation dataset key')
    continue_parser.add_argument('--epoch', type=int, nargs='+',
                                 help='learning epoch', required=True)

    continue_parser.set_defaults(func=command_continue)

    # predict parser
    predict_parser = subparser.add_parser('predict', help='continue help')
    predict_parser.add_argument('model_key', type=str,
                                help='model key to predict')
    predict_parser.add_argument('org_dataset_path', type=str,
                                help='original dataset directory path')
    predict_parser.add_argument('save_dataset_path', type=str,
                                help='save predicted image directory path')
    predict_parser.add_argument('csv_dir_path', type=str,
                                help='output predict result directory path. save csv file in directory.')

    predict_parser.set_defaults(func=command_predict)

    # increase parser
    increase_parser = subparser.add_parser('increase', help='increase help')
    increase_parser.add_argument('org_dataset_path', type=str,
                                 help='original data directory path')
    increase_parser.add_argument('save_dataset_path', type=str,
                                 help='save increased data directory path')
    increase_parser.add_argument('increase_num', type=int,
                                 help='the number of increasing data per tag')

    increase_parser.set_defaults(func=command_increase)

    # divide parser
    divide_parser = subparser.add_parser('divide', help='divide help')
    divide_parser.add_argument('org_dataset_path', type=str,
                               help='original data directory path')
    divide_parser.add_argument('save_dataset_path', type=str,
                               help='save divided data directory path')
    divide_parser.add_argument('divide_num', type=int,
                               help='the number of data dividing into validation')

    divide_parser.set_defaults(func=command_divide)

    return parser


def get_store():
    model_store = context.get(
        context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore
    dataset_store = context.get(
        context.CONTEXT_KEY_DATASET_STORE, DatasetStore)  # type: DatasetStore

    return model_store, dataset_store


def get_service(model_store: ModelStore, dataset_store: DatasetStore):
    return BottomLengthClassificationService(model_store, dataset_store)


def make_row_data(img_path: str, correct_label: str, label_params: dict):
    csv_row = []
    csv_row.append(img_path)

    csv_row.append(correct_label)
    for label_param in label_params['labels']:
        csv_row.append(label_param['name'])
        csv_row.append(label_param['acc'])

    return csv_row


def command_learn(train_dataset_key: str=None,
                  validation_dataset_key: str=None,
                  class_num: int=None, epoch: int=None, **args):
    model_store, dataset_store = get_store()
    service = get_service(model_store, dataset_store)

    model_container = BottomLengthModelContainerAtResNet(class_num)
    new_model_key = service.learn(model_container,
                                  model_container.get_model_key() + '-' + str(class_num) + 'class',
                                  train_dataset_key,
                                  validation_dataset_key,
                                  (3, 300, 200), epoch,
                                  is_data_aug=True, aug_rate=3)


def command_continue(model_key: str=None, train_dataset_key: str=None,
                     validation_dataset_key: str=None,
                     epoch: List=None, **args):
    model_store, dataset_store = get_store()
    service = get_service(model_store, dataset_store)

    for e in epoch:
        LOGGER.debug('learn start:%s', model_key)
        model_key = service.continue_learning(model_key,
                                              train_dataset_key,
                                              validation_dataset_key,
                                              (300, 200, 3), e,
                                              is_data_aug=True, aug_rate=3)
        tf.clear_session()


@logging_time
def command_predict(model_key: str=None, org_dataset_path: str=None,
                    save_dataset_path: str=None,
                    csv_dir_path: str=None, **args):
    import csv

    model_store, dataset_store = get_store()
    model_container = model_store.load(model_key, use_all_in_one=False)
    model = model_container.get_model()

    service = BottomLengthClassificationService(model_store, dataset_store)

    label = ['above_the_knee', 'asymmetry', 'below_the_knee', 'cropped',
             'full_length', 'long', 'micromini', 'midi', 'mini']

    processor = ClassificationProcessor(label)

    if not os.path.exists(csv_dir_path):
        os.mkdir(csv_dir_path)
    csv_name = 'bottom_length-predict' + dt.now().strftime('-%Y%m%d%H%M%S') + '.csv'
    csv_file_path = os.path.join(csv_dir_path, csv_name)
    with open(csv_file_path, mode='w', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        counter = 0
        for root, dirs, files in os.walk(org_dataset_path):
            for file in files:
                img_path = os.path.join(root, file)
                root_dir = os.path.basename(root)
                correct_label = root_dir if root_dir in label else None
                if os.path.isfile(img_path):
                    img = Image.open(img_path).resize(
                        (200, 300))  # type: Image.Image
                    img = image_utils.to_array_and_reshape(img)
                    result = model.predict_on_batch(img)

                    label_params, predict_labels = processor.process(result)

                    # save image in predict label directory
                    dist_sub_dir = os.path.join(
                        save_dataset_path, predict_labels[0]['name'])
                    if not os.path.exists(dist_sub_dir):
                        os.mkdir(dist_sub_dir)
                    shutil.copy(img_path, dist_sub_dir)
                    counter += 1

                    # output predict result to csv file
                    writer.writerow(make_row_data(
                        img_path, correct_label, label_params))

        LOGGER.info('predict count:%s', str(counter))
        LOGGER.info('predict result output to %s', csv_file_path)


def command_increase(org_dataset_path: str=None,
                     save_dataset_path: str=None,
                     increase_num: int=None, **args):
    image_utils.increase_img_data(org_dataset_path,
                                  save_dataset_path,
                                  increase_num,
                                  generator_option={
                                      'vertical_flip': False,
                                      'horizontal_flip': True,
                                      'width_shift_range': 0.2,
                                      'height_shift_range': 0.1,
                                      'zoom_range': 0.2
                                  },
                                  flow_option={
                                      'target_size': (300, 200)
                                  })


def command_divide(org_dataset_path: str=None,
                   save_dataset_path: str=None,
                   divide_num: int=None, **args):
    image_utils.divide_to_validation(org_dataset_path,
                                     save_dataset_path,
                                     divide_num)


if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()

    # app initialize
    context.Initializer().initialize()
    LOGGER.debug(vars(args))
    # execute subcommand.
    args.func(**vars(args))

#!/usr/bin/env python35
# -*- coding: utf-8 -*-


import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np

from app.core.log import logging_time
from typing import List, Tuple

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

LOGGER = logging.getLogger(__name__)

# const
IMG_SIZE = (600, 400)  # height, width
FCN_CLASSES = 16


def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(description='Process')
    subparser = parser.add_subparsers()

    # continue parser
    continue_parser = subparser.add_parser('continue', help='continue help')
    continue_parser.add_argument(
        'model_key', type=str, help='model key to continue')
    continue_parser.add_argument(
        'dataset_path', type=str, help='dataset store path')
    continue_parser.add_argument(
        '--is_trainable', dest='is_trainable', action='store_true')
    continue_parser.set_defaults(is_trainable=False)
    continue_parser.add_argument(
        '--epoch', type=int, nargs='+', help='learning epoch', required=True)

    continue_parser.set_defaults(func=command_continue)

    # learn parser
    learn_parser = subparser.add_parser('learn', help='learn help')
    learn_parser.add_argument('model_num', choices={
                              '50', '32', '16', '8'}, type=str, help='fcn model help')
    learn_parser.add_argument('dataset_path', type=str,
                              help='dataset store path')
    learn_parser.add_argument('epoch', type=int, help='learning epoch')
    learn_parser.add_argument('prev_model_key', type=str,
                              help='prev model key. required in the case of fcn 16s or 8s.')
    learn_parser.add_argument(
        '--is_trainable', dest='is_trainable', action='store_true')
    learn_parser.set_defaults(is_trainable=False)

    learn_parser.set_defaults(func=command_learn)

    # predict parser
    predict_parser = subparser.add_parser('predict', help='continue help')
    predict_parser.add_argument(
        'model_key', type=str, help='model key to predict')
    predict_parser.add_argument(
        'image_path', type=str, help='image sotre path. directory or file.')

    predict_parser.set_defaults(func=command_predict)

    return parser


def get_fcn_service() -> 'FcnService':
    from app.models.manager import ModelStore
    from app.services import fcn
    from app.services.fcn import FcnService

    model_store = context.get(context.CONTEXT_KEY_MODEL_STORE, ModelStore)
    return FcnService(model_store)


def get_predict_result_save_path(dir_name: List[str], file_name: str) -> Path:
    save_dir = Path(os.path.join(context.get(
        'fcn.dir', str), 'image', *dir_name))

    LOGGER.info('save dir for predict result:%s', str(save_dir))
    if not save_dir.exists():
        os.makedirs(str(save_dir))

    save_path = save_dir.joinpath(file_name + '-predict.jpg')
    index = 1
    while save_path.exists():
        save_path = save_dir.joinpath(
            file_name + '-predict' + str(index) + '.jpg')
        index += 1
    return Path(save_path)


def command_learn(model_num: str=None, dataset_path: str=None,
                  epoch: int=None, prev_model_key: str=None,
                  is_trainable: bool=False, **args):

    fcn_service = get_fcn_service()  # type: FcnService

    if model_num == '50':
        fcn_service.learn_resnet(dataset_path,
                                 epoch, IMG_SIZE[1], IMG_SIZE[0],
                                 FCN_CLASSES)

    elif model_num == '32':
        fcn_service.learn_using_vgg16_weights(dataset_path,
                                              epoch, IMG_SIZE[1], IMG_SIZE[0],
                                              FCN_CLASSES,
                                              set_upsample_weights=True)
    elif model_num == '16':
        fcn_service.learn_using_fcn32s_weights(dataset_path,
                                               epoch, IMG_SIZE[1], IMG_SIZE[0],
                                               FCN_CLASSES, prev_model_key,
                                               is_trainable=is_trainable)

    elif model_num == '8':
        fcn_service.learn_using_fcn16s_weights(dataset_path,
                                               epoch, IMG_SIZE[1], IMG_SIZE[0],
                                               FCN_CLASSES, prev_model_key,
                                               is_trainable=is_trainable)

    else:
        LOGGER.error(
            'unexpected model_num. expected:32,16,8. actula:%s', model_num)


def command_continue(model_key: str=None, dataset_path: str =None,
                     is_trainable: bool =None, epoch: List=None,
                     **args):
    '''
    保存済のモデルを流用する学習処理
    arguments
        size: (height, width)
    '''
    from keras.backend import tensorflow_backend as tf

    LOGGER.info('args:%s', args)
    fcn_service = get_fcn_service()

    for e in epoch:
        LOGGER.info('use model key: %s', model_key)
        model_key, class_name = fcn_service.learn_continuous(dataset_path,
                                                             e, IMG_SIZE,
                                                             FCN_CLASSES, model_key, 0,
                                                             is_override_trainable=is_trainable,
                                                             set_upsample_weights=False)

        LOGGER.info('learn end. model_key:%s , model:%s',
                    model_key, class_name)
        tf.clear_session()

    return model_key


@logging_time
def command_predict(model_key: str=None, image_path: str=None, **args):
    '''
    評価処理
    '''
    from app.constants.color_palette import LabelEnum
    from app.models.fcn import SegmentationProcessor

    label = LabelEnum.of(FCN_CLASSES)
    predict_processor = SegmentationProcessor(IMG_SIZE)

    fcn_service = get_fcn_service()
    image_path = Path(image_path)

    if image_path.is_file():
        LOGGER.info('process single file.')
        LOGGER.info('file:%s', image_path)
        # process file
        result, crf_result = fcn_service.predict(model_key,
                                                 image_path,
                                                 IMG_SIZE,
                                                 FCN_CLASSES,
                                                 print_summary=False)

        file_name = image_path.stem
        label.to_image(result).save(
            get_predict_result_save_path(['single'], file_name))
        label.to_image(crf_result).save(
            get_predict_result_save_path(['single'], file_name + '-crf'))

    else:
        LOGGER.info('process multi file.')
        # process dir
        model = model_store.load(model_key, use_all_in_one=False).get_model()

        file_count = 0
        for i, path in enumerate(image_path.glob('**/*.jpg')):
            LOGGER.info('file:%s', path)
            img = fcn_service.preprocces_for_predict(path, size)
            result = model.predict_on_batch(img)
            result = predict_processor.process(result)

            file_name = path.stem
            parent_dir_name = path.parent.name
            label.to_image(result).save(
                get_predict_result_save_path(['2015', str(parent_dir_name)], file_name))
            file_count += 1


if __name__ == '__main__':

    # get cmd args
    args = define_arg_parser().parse_args()

    from app import context
    # app initialize
    context.Initializer().initialize()

    # set full format logging for numpy.
    np.set_printoptions(threshold=np.inf)

    # execute subcommand.
    args.func(**vars(args))

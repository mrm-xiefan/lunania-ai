#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from json_tricks import dumps, dump
LOGGER = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def define_arg_parser() -> argparse.ArgumentParser:
    '''
    コマンドラインで使用する引数を定義する。
    '''
    parser = argparse.ArgumentParser(description='image tagging.')
    parser.add_argument('input', help="input image file path.")
    parser.add_argument(
        '--dist_img_dir', default=None, help='output detection area plot image.')

    parser.set_defaults(func=command_tagging)
    return parser


def command_tagging(input: str=None,
                    dist_img_dir: str=None,
                    **args):

    from app.models.manager import ModelStore
    from app.services.tag import TagClassificationService

    model_store = context.get(
        context.CONTEXT_KEY_MODEL_STORE, ModelStore)  # type: ModelStore

    file_name, _ = os.path.splitext(os.path.basename(input))

    # run tagging
    tag_service = TagClassificationService(model_store)
    result = tag_service.get_tags(
        load_img(input),
        output_detection=dist_img_dir,
        output_file_name=file_name)
    with open('./result.json', 'w', encoding='utf-8') as f:
        dump(result, f, indent=2, primitives=True, ensure_ascii=False)

if __name__ == '__main__':
    # get cmd args
    args = define_arg_parser().parse_args()

    # keras import
    from keras.preprocessing.image import load_img

    # app initialize
    from app import context
    context.Initializer().initialize()

    args.func(**vars(args))

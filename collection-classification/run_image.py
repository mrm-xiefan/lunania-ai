#!/usr/bin/env python35
# -*- coding: utf-8 -*-

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import glob
    import logging
    import numpy as np
    from PIL import Image
    import argparse
    LOGGER = logging.getLogger(__name__)

    from app import context
    # app initialize
    context.Initializer().initialize()
    app_config = context.get(
        context.CONTEXT_KEY_APP_CONFIG, context.ApplicationConfig)
    np.set_printoptions(threshold=np.inf)

    from app.utils import image_utils
    from app.constants.color_palette import LabelEnum

    FCN_DATA_DIR = app_config.get(['fcn', 'dir'])
    CROP_DATA_DIR = os.path.join(FCN_DATA_DIR, 'crop')

    _fcn_classes = 5

    # TODO 入力ファイルパスの指定方法を修正
    _original_file_name = '_MON0201.jpg'
    _segmentation_file_name = '5_MON0201.png'
    _segmentation_npy_file_name = '5_MON0201.npy'
    _origin_data = Image.open(os.path.join(
        FCN_DATA_DIR, 'collection_data', _original_file_name))
    _segmentation_npy_data = np.load(os.path.join(
        FCN_DATA_DIR, 'collection_output', _segmentation_npy_file_name))

    label_colors = LabelEnum.of(_fcn_classes).get_colors()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('process', choices=['crop', 'fill', 'mark'])
    args = parser.parse_args()

    # 分類ラベルの範囲毎に、それぞれの領域でトリミングした画像を保存(複数ファイル)
    if args.process in {'crop'}:
        # インスタンス生成時、_segmentation_npy_dataを用いて、
        # 各ラベルに対する存在領域の範囲を取得する
        processor = image_utils.LabelExtentProcessor(
            _segmentation_npy_data, label_colors, margin_size=(5, 5), aspect_ratio=(3, 2))
        org_output_dir = os.path.join(CROP_DATA_DIR, '5_collection')
        npy_output_dir = os.path.join(CROP_DATA_DIR, '5_collection_npy')
        seg_output_dir = os.path.join(CROP_DATA_DIR, '5_collection_seg')

        for label in processor.get_exists_labels():
            LOGGER.info('cropping index %s start', label)
            # 元画像をトリミングした画像を保存
            output_image = processor.crop(label, _origin_data)

            save_dir = os.path.join(org_output_dir, str(label))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            output_image.save(os.path.join(
                save_dir, '5' + _original_file_name))
            LOGGER.info('save %s in %s' %
                        (save_dir, '5' + _original_file_name))

            # トリミングした画像に対する、ピクセル毎の分類ラベルを値に持つNumpy配列を保存
            crop_npy = processor.crop_npy_data(
                label, _segmentation_npy_data)

            save_dir = os.path.join(npy_output_dir, str(label))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            np.save(os.path.join(
                save_dir, _segmentation_npy_file_name), crop_npy)
            LOGGER.info('save %s in %s' %
                        (save_dir, _segmentation_npy_file_name))

            # トリミング後のセグメンテーション画像を保存
            label_colors = LabelEnum.of(_fcn_classes).get_colors()
            seg_image = np.zeros((crop_npy.shape[0], crop_npy.shape[1], 3))
            for h, tmp in enumerate(crop_npy):
                for w, label_index in enumerate(tmp):
                    seg_image[h, w] = label_colors[int(label_index)]

            save_dir = os.path.join(seg_output_dir, str(label))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            Image.fromarray(np.uint8(seg_image)).save(os.path.join(
                save_dir, _segmentation_file_name))

            LOGGER.info('save %s in %s' %
                        (save_dir, _segmentation_file_name))

            LOGGER.info('index %s is cropped', label)

    # 元画像において、分類ラベルがbackgroundの部分を塗りつぶす
    elif args.process in {'fill'}:
        processor = image_utils.LabelFillingProcessor()

        filling_image = processor.fill(_origin_data, _segmentation_npy_data)
        filling_image.save(os.path.join(
            FCN_DATA_DIR, 'fill', _original_file_name))

    # 分類ラベルの範囲毎に、それぞれの領域を矩形で示した画像を保存(単一ファイル)
    elif args.process in {'mark'}:
        LOGGER.info('marking image by label is start')
        output_data_dir = os.path.join(OUTPUT_DATA_DIR, 'mark')
        img = processor.mark_all(_origin_data)
        # TODO 保存ファイル名のルール決定、反映
        img.save(os.path.join(output_data_dir, 'test_2.jpg'))
        LOGGER.info('save %s in %s' % ('test_2.jpg', output_data_dir))
        LOGGER.info('marking image by label is end')

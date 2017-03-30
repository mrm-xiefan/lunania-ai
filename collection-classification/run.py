#!/usr/bin/env python35
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import os
    import logging
    import argparse
    LOGGER = logging.getLogger(__name__)

    from keras.utils import np_utils
    from keras.models import Sequential

    # app initialize
    from app import context
    context.Initializer().initialize()

    from app import learning, prediction
    from app.datasets.manager import DatasetStore
    from app.models.manager import ModelStore
    from app.models.color import ColorClassification

    LOGGER.info('cwd:%s', os.getcwd())
    dataset_store = context.get(
        context.CONTEXT_KEY_DATASET_STORE, DatasetStore)
    model_store = context.get(context.CONTEXT_KEY_MODEL_STORE, ModelStore)

    colorClazz = ColorClassification(model_store)

    learning_dataset = dataset_store.load('color-aug', 150, 150)
    validation_dataset = dataset_store.load('color', 150, 150)

    def fit_func(model: Sequential):
        return model.fit_generator(learning_dataset, len(learning_dataset.classes), 30, validation_data=validation_dataset, nb_val_samples=250)

    colorClazz.learn('color-aug', (150, 150, 3), fit_func)

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('process', choices=['learn', 'predict', 'fcn'])
    # parser.add_argument('dataset_key')
    # parser.add_argument('--is_npy_use', action='store_const', const=False, default=True)
    # parser.add_argument('--image_path')

    # args = parser.parse_args()
    # if args.process in {'learn'}:
    #     learning.learn(args.dataset_key, is_npy_use=args.is_npy_use)
    # elif args.process in {'predict'}:
    #     import shutil
    #     save_base_dir = '/home/hiroki.endo/work/data/test/vetements-result'
    #     path = '/home/hiroki.endo/work/data/test/vetements'
    #     for file_name in os.listdir(path):
    #         file = os.path.join(path, file_name)
    #         clazz, label = prediction.predict(args.dataset_key, file)
    #         save_dir = os.path.join(save_base_dir, label)
    #         try:
    #             os.makedirs(save_dir)
    #         except FileExistsError:
    #             pass

    #         shutil.copy2(file, os.path.join(save_dir, file_name))

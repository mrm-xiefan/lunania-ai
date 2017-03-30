#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import os

import h5py
import numpy as np

from app import context

LOGGER = logging.getLogger(__name__)


def create_upsample_filter(size) -> np.ndarray:
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


class NullLogger():

    def info(msg, *args, **kwargs):
        pass

    def debug(msg, *args, **kwargs):
        pass


def interp_upsample_filter(layer, filter_scale=1, verbose=False):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    if verbose:
        LOG = LOGGER
    else:
        LOG = NullLogger()

    weight = layer.get_weights()
    LOG.debug(weight)
    if weight:
        weight0 = weight[0]
        weight1 = weight[1]

    else:
        input_size = layer.input_shape[3]
        weight0 = np.zeros(
            (layer.nb_row, layer.nb_col, input_size, input_size))
        weight1 = np.zeros((input_size))
        layer.trainable_weights = [
            np.zeros((layer.nb_row, layer.nb_col, input_size, input_size))]
        layer.non_trainable_weights = [np.zeros((input_size))]

    LOG.info('layer input_shape:%s', layer.input_shape)
    LOG.info('weight0: %s', weight0.shape)
    LOG.info('weight1: %s', weight1.shape)

    r, c, input, output = weight0.shape
    # if m != k and k != 1:
    #     LOGGER.error('input + output channels need to be the same or |output| == 1')
    #     raise
    if r != c:
        LOGGER.error('filters need to be square')
        raise

    filter = create_upsample_filter(r)
    stack_filter = filter
    if output == 1:
        stack_filter = filter.reshape((r, c, 1, 1))
        LOG.debug('stack_filter shape:%s', stack_filter.shape)
    else:
        for i in range(output - 1):
            stack_filter = np.dstack((stack_filter, filter))

    stack_filter = np.array([stack_filter for i in range(input)])

    weight0 = stack_filter.reshape(r, c, input, output)
    LOG.info('weight0.shape:%s', weight0.shape)
    LOG.info('weight0:%s', weight0)
    weight1 = np.zeros(output)
    layer.set_weights([weight0, weight1])


def set_vgg16_weights(model, is_trainable=True) -> None:
    join = os.path.join

    weight_file_path = join(context.get('model.dir', str),
                            context.get('model.vgg16.path.weight', str))

    with h5py.File(weight_file_path) as weight_file:
        generator = _conv_weights_generator(weight_file)

        for layer in model.layers:
            LOGGER.debug('target layer:%s', layer.name)
            if 'conv_' in layer.name:
                weights = generator.__next__()
                layer.set_weights(weights)

                LOGGER.debug('set weights in %s layer' % layer.name)
                LOGGER.debug(str(is_trainable))
                layer.trainable = is_trainable

    return model


def _conv_weights_generator(file):
    while 1:
        for k in range(file.attrs['nb_layers']):
            g = file['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])]
            # convolution層の重みのみ取り出す
            if len(weights) > 0:
                # TensorFlowの形式に重みパラメータの形式を変換
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
                yield(weights)

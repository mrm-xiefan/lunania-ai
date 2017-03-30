#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging
import numpy as np

from flask.json import JSONEncoder

LOGGER = logging.getLogger(__name__)


class CustomJSONEncoder(JSONEncoder):

    def default(self, obj):
        LOGGER.debug('start.')
        if isinstance(obj, np.ndarray):
            LOGGER.debug('np.ndarray')
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            LOGGER.debug('np.core.multiarray.scalar')
            return obj.tolist()

        return JSONEncoder.default(self, obj)

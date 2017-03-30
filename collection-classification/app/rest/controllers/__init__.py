#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import sys
import logging
import glob
import importlib
import re
import os

LOGGER = logging.getLogger(__name__)

_BLUEPRINT_VARIABLE = 'app'


def register(application):
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    _self = sys.modules[__name__]
    LOGGER.info('LOAD module:%s', _self)

    paths = glob.glob(os.path.join(this_file_path, '*.py'))
    for py_file in paths:
        mod_name = os.path.splitext(os.path.basename(py_file))[0]
        if re.search(".*__init__.*", mod_name) is None:
            mod = importlib.import_module(__name__ + "." + mod_name)
            if hasattr(mod, _BLUEPRINT_VARIABLE):
                LOGGER.debug('add flask module:%s', mod)
                application.register_blueprint(mod.app)

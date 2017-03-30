#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import logging

LOGGER = logging.getLogger(__name__)


class UnsupportedOperationError(Exception):

    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self.__class__.__name__ + ':{}'.format(self._msg)

#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime as dt
import time
from functools import wraps


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class DailyRotateFileHandler(logging.FileHandler):

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        root, ext = os.path.splitext(filename)
        filename = root + dt.now().strftime('.%Y-%m-%d') + ext

        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)


def logging_time(func: callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        __logger = logging.getLogger(func.__name__)
        __logger.debug('start.')
        __start = time.time()
        ret = func(*args, **kwargs)
        __logger.debug('end. psec:%s', time.time() - __start)
        return ret
    return wrapper

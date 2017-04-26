import os
import config


def lock():
    if os.path.exists(config.lock_file):
        exit('Previous process is not yet finished.')
    lock_file = open(config.lock_file, 'w')
    lock_file.write(str(os.getpid()))
    lock_file.close()


def unlock():
    if os.path.exists(config.lock_file):
        os.remove(config.lock_file)

def error(code):
    print({'error': code})


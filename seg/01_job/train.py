import os
import utils
import config
import traceback
import argparse
import logging.config
from luna import LunaExcepion


logging.config.fileConfig("logging.conf")
logger = logging.getLogger()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg', help='vgg or resnet')
    parser.add_argument('--data', required=True, help='dataset path')
    parser.add_argument('--epoch', default=1, help='epoches')
    return parser.parse_args()

if __name__ == '__main__':
    try:
        logger.info("------ start ------")
        utils.lock()

        args = parse_args()
        if not os.path.exists(args.data):
            raise LunaExcepion(config.inputerr)
        if (args.model != 'vgg' and args.model != 'resnet'):
            raise LunaExcepion(config.inputerr)

        from fcn import Fcn
        import numpy as np

        if not os.path.exists(config.model_dir):
            os.mkdir(config.model_dir)

        fcn = Fcn(args.model)
        fcn.train(args.data, args.epoch)
        logger.info("------ end ------")

    except (KeyboardInterrupt, SystemExit):
        utils.unlock()
        utils.error(config.syserr)
    except LunaExcepion as e:
        utils.error(e.value)
        if (e.value == config.locked):
            exit()
            logger.info("------ end ------")
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        utils.error(config.syserr)
    utils.unlock()
    logger.info("------ end ------")

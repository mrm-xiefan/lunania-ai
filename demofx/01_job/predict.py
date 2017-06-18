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

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', required=True, help='input an image to predict')
#     parser.add_argument('--model', default='vgg-22class-200epoch-20170609053802', help='trained model in 90_result')
#     return parser.parse_args()


if __name__ == '__main__':
    try:
        logger.info("------ start ------")
        utils.lock()

        # args = parse_args()
        # if not os.path.exists(args.image):
        #     raise LunaExcepion(config.inputerr)

        # here write some logic

        # return json to website.
        print({"error": "", "data": ""})

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

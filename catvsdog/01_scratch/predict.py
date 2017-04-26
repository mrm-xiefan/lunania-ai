import os
import utils
import config
import traceback
import argparse
from luna import LunaExcepion

from keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='1', help='1: small cnn. 2: top trained. 3: fine tuned')
    parser.add_argument('--image', required=True, help='input an image to predict')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        utils.lock()

        args = parse_args()
        if not os.path.exists(args.image):
            raise LunaExcepion(config.inputerr)

        if args.model == '1':
            print(args.model)
        elif args.model == '2':
            print(args.model)
        elif args.model == '3':
            print(args.model)
        else:
            raise LunaExcepion(config.inputerr)
    except (KeyboardInterrupt, SystemExit):
        utils.unlock()
        utils.error(config.syserr)
    except LunaExcepion as e:
        utils.error(e.value)
    except Exception as e:
        utils.error(config.syserr)
        """
        print(e)
        print(traceback.format_exc())
        """
    utils.unlock()


import os
import utils
import config
import traceback
import argparse
import logging.config
from luna import LunaExcepion


logging.config.fileConfig("logging.conf")
logger = logging.getLogger()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='input an image to predict')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        logger.info("------ start ------")
        utils.lock()

        args = parse_args()
        if not os.path.exists(args.image):
            raise LunaExcepion(config.inputerr)

        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        from keras.preprocessing import image
        import numpy as np

        # 学習済みのVGG16をロード
        # 構造とともに学習済みの重みも読み込まれる
        model = VGG16(weights='imagenet')

        # 画像を読み込んで4次元テンソルへ変換
        img = image.load_img(args.image, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # クラスを予測
        # 入力は1枚の画像なので[0]のみ
        preds = model.predict(preprocess_input(x))

        # 予測確率が高いトップ5を出力
        results = decode_predictions(preds, top=5)[0]
        data = []
        for result in results:
            data.append({"name": result[1], "percentage": '%.10f' % (result[2] * 100)});
        print({"error": "", "data": data})
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

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

        from keras.models import Sequential, Model
        from keras.models import load_model
        from keras.preprocessing import image
        from keras.applications.vgg16 import VGG16
        from keras.layers import Input, Activation, Dropout, Flatten, Dense
        import numpy as np

        classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
                   'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
                   'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
                   'Windflower', 'Pansy']
        nb_classes = len(classes)

        # VGGのモデルをロード、FC層は自分の学習済みのモデルにするので、不要
        input_tensor = Input(shape=(config.img_height, config.img_width, config.channels))
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        # fc層のガラを作る
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(nb_classes, activation='softmax'))
        # 二つのモデルを結合する
        model = Model(inputs=vgg16_model.input, outputs=top_model(vgg16_model.output))
        # 学習済みの重みをロード
        model.load_weights(os.path.join(config.result_dir, 'finetuning_weights.h5'))
        # compile
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # 画像を読み込んで4次元テンソルへ変換
        img = image.load_img(args.image, target_size=(config.img_height, config.img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
        x = utils.preprocess_images(x)
        # クラスを予測
        # 入力は1枚の画像なので[0]のみ
        pred = model.predict(x)[0]

        # 予測確率が高いトップ5を出力
        top = 5
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i], pred[i]) for i in top_indices]
        data = []
        for x in result:
            data.append({"name": x[0], "percentage": '%.10f' % (x[1] * 100)})
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

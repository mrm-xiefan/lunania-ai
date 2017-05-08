import os
import utils
import config
import traceback
import logging.config
from luna import LunaExcepion


logging.config.fileConfig("logging.conf")
logger = logging.getLogger()


if __name__ == '__main__':
    try:
        logger.info("------ start ------")
        utils.lock()


        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D
        from keras.layers import Activation, Dropout, Flatten, Dense
        from keras.preprocessing.image import ImageDataGenerator
        from keras.utils.vis_utils import plot_model
        import numpy as np


        if not os.path.exists(config.result_dir):
            os.mkdir(config.result_dir)

        # モデルを構築
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        #model.summary()
        plot_model(model, to_file='model.png')

        # 訓練データとバリデーションデータを生成するジェネレータを作成
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(
            config.train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )
        validation_generator = test_datagen.flow_from_directory(
            config.validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )
        print(train_generator.class_indices)

        # 訓練
        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(np.floor(2000/32)),
            epochs=50,
            validation_data=validation_generator,
            validation_steps=int(np.floor(800/32))
        )
        utils.plot_history(history)

        # 結果を保存
        model.save(os.path.join(config.result_dir, 'scratch_model.h5'))
        model.save_weights(os.path.join(config.result_dir, 'scratch_weights.h5'))
        utils.save_history(history, os.path.join(config.result_dir, 'scratch_history.txt'))

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

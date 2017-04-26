import os
import utils
import config
import traceback
from luna import LunaExcepion

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    try:
        utils.lock()

        if not os.path.exists(config.result_dir):
            os.mkdir(config.result_dir)

        # モデルを構築
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
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
            
        # 訓練
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=2000,
            nb_epoch=10,
            validation_data=validation_generator,
            nb_val_samples=800
        )

        # 結果を保存
        model.save(os.path.join(config.result_dir, 'scratch_model.h5'))
        model.save_weights(os.path.join(config.result_dir, 'scratch_weights.h5'))
        
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


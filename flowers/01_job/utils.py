import os
import config
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from luna import LunaExcepion


logger = logging.getLogger()


def preprocess_images(images):
    # 'RGB'->'BGR'
    images = images[:, :, ::-1]
    # Zero-center by mean pixel
    images[:, :, 0] -= 103.939
    images[:, :, 1] -= 116.779
    images[:, :, 2] -= 123.68
    return images

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('acc.png')

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig('loss.png')

def lock():
    if os.path.exists(config.lock_file):
        raise LunaExcepion(config.locked)
    lock_file = open(config.lock_file, 'w')
    lock_file.write(str(os.getpid()))
    lock_file.close()


def unlock():
    if os.path.exists(config.lock_file):
        os.remove(config.lock_file)

def error(code):
    logger.error(code)
    print({'error': code})


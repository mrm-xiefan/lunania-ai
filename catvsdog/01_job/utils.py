import os
import config
import matplotlib.pyplot as plt

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

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


from os.path import join as join_path
import os


abspath = os.path.dirname(os.path.abspath(__file__))
lock_file = '/home/ai/lock'
vgg16_weights_file = '/home/ai/Models/vgg16/model.h5'
model_dir = join_path(abspath, '..', '90_result')
predict_dir = '/opt/homepage-miraimon/public/upload/'

img_height = 500
img_width = 500
classes = 22

syserr = 'S001'
inputerr = 'B001'
locked = 'B002'
unsupport = 'B003'


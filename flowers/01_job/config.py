from os.path import join as join_path
import os


abspath = os.path.dirname(os.path.abspath(__file__))
lock_file = '/home/ai/lock'
data_dir = join_path(abspath, '..', '99_data')
train_dir = join_path(data_dir, 'train')
validation_dir = join_path(data_dir, 'validation')
result_dir = join_path(abspath, '..', '90_result')

img_height = 150
img_width = 150
channels = 3

syserr = 'S001'
inputerr = 'B001'
locked = 'B002'


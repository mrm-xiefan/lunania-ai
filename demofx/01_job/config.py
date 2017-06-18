from os.path import join as join_path
import os


abspath = os.path.dirname(os.path.abspath(__file__))
lock_file = 'lock'
train_dir = join_path(abspath, '..', '99_data')
result_dir = join_path(abspath, '..', '90_result')


syserr = 'S001'
inputerr = 'B001'
locked = 'B002'
unsupport = 'B003'


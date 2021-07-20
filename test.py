import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__获取执行文件相对路径，整行为取上一级目录
sys.path.append(BASE_DIR)


import mindspore as ms
from cv import test_cv


test_cv.func()
print(sys.path)


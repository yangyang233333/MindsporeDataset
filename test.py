import os, sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # __file__获取执行文件相对路径，整行为取上一级目录
# sys.path.append(BASE_DIR)


import mindspore as ms

# print(sys.path)


# print(__file__)
# print(os.path.abspath(__file__))
#
# print(os.path.dirname(os.path.abspath(__file__)))
os.system("wget -P dataset/test http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz")


import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype
import glob
import cv2

# 数据集下载链接
resources1 = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

# 数据集内的所有文件
resources2 = ['emnist-balanced-mapping.tx',
              'emnist-balanced-test-images-idx3-ubyte.gz',
              'emnist-balanced-test-labels-idx1-ubyte.gz',
              'emnist-balanced-train-images-idx3-ubyte.gz',
              'emnist-balanced-train-labels-idx1-ubyte.gz',

              'emnist-byclass-mapping.txt',
              'emnist-byclass-test-images-idx3-ubyte.gz',
              'emnist-byclass-test-labels-idx1-ubyte.gz',
              'emnist-byclass-train-images-idx3-ubyte.gz',
              'emnist-byclass-train-labels-idx1-ubyte.gz',

              'emnist-bymerge-mapping.txt',
              'emnist-bymerge-test-images-idx3-ubyte.gz',
              'emnist-bymerge-test-labels-idx1-ubyte.gz',
              'emnist-bymerge-train-images-idx3-ubyte.gz',
              'emnist-bymerge-train-labels-idx1-ubyte.gz',

              'emnist-digits-mapping.txt',
              'emnist-digits-test-images-idx3-ubyte.gz',
              'emnist-digits-test-labels-idx1-ubyte.gz',
              'emnist-digits-train-images-idx3-ubyte.gz',
              'emnist-digits-train-labels-idx1-ubyte.gz',

              'emnist-letters-mapping.txt',
              'emnist-letters-test-images-idx3-ubyte.gz',
              'emnist-letters-test-labels-idx1-ubyte.gz',
              'emnist-letters-train-images-idx3-ubyte.gz',
              'emnist-letters-train-labels-idx1-ubyte.gz',

              'emnist-mnist-mapping.txt',
              'emnist-mnist-test-images-idx3-ubyte.gz',
              'emnist-mnist-test-labels-idx1-ubyte.gz',
              'emnist-mnist-train-images-idx3-ubyte.gz',
              'emnist-mnist-train-labels-idx1-ubyte.gz']


class _EMNIST:
    """加载EMIST的辅助类"""

    def __init__(self, root="./dataset/emnist", is_train=True, download=False):
        """
        :param root:
        :param is_train:
        :param download:
        """
        self.root = root

        # 检查数据集文件是否完整
        self.check_exist()

        if is_train:
            self.data = None
            self.label = None
        else:
            self.data = None
            self.label = None

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_image(self):

        pass

    def read_label(self):

        pass

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip下载数据集并且解压好")
                raise RuntimeError()

    # def download(self):
    #     """下载数据集"""
    #     # Win10 和 Linux的下载方法不一样，有点麻烦，暂时搁置
    #     raise RuntimeError("请去https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip下载数据集并且解压好")


def emnist():
    pass


if __name__ == '__main__':
    base_path = r'E:\MindsporeVision\gzip'

    file_name = base_path + r'\emnist-balanced-test-labels-idx1-ubyte.gz'
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    # struct.unpack()

    offset = struct.calcsize('>IIII')
    # img_num = head[1]  # 图片数
    # width = head[2]  # 宽度
    # height = head[3]  # 高度
    print(head)

    # content = struct.unpack_from('>784B', file_content, offset)
    # imgs_array = np.array(content).reshape((1, 28, 28)).transpose((1, 2, 0)) / 255
    # print(imgs_array.shape)
    # cv2.imshow('x', imgs_array)
    # cv2.waitKey(0)
    #
    # print(imgs_array)

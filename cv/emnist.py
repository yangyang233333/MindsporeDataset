import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype

# 数据集下载链接
resourse = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


class _EMNIST:
    """加载EMIST的辅助类"""

    def __init__(self, root="dataset/emnist", usage="train", download=False):
        self.root = root
        if download:
            self.download()



    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def check_exist(self):
        """检查数据集文件是否存在"""
        pass

    def download(self):
        """下载数据集"""
        pass


def emnist():
    pass


if __name__ == '__main__':
    os.system("cmd")
    os.system("wget -P ./dataset/test http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz")
    pass



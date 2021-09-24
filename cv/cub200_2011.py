# -*- coding: UTF-8 -*-
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype
import glob
import cv2
import gzip
import shutil

# 数据集下载链接
resources1 = 'http://www.vision.caltech.edu/visipedia/CUB-200-2011.html'

# 数据集内的所有文件
resources2 = ['attributes', 'bounding_boxes.txt', 'classes.txt', 'images', 'images.txt', 'image_class_labels.txt',
              'parts', 'README', 'train_test_split.txt']


class Cub200Generator:
    def __init__(self, root, is_training=True):
        """

        :param root:
        :param is_training:
        """

        self.root = root
        self.is_training = is_training  # True表示训练集，False表示测试集

        # 检查数据集是否完整
        self.check_exist()

        id2data = np.genfromtxt(self.root + r'/images.txt', delimiter=' ', dtype='str')

        id2label = np.genfromtxt(self.root + r'/image_class_labels.txt', delimiter=' ', dtype='str')

        id2istraining = np.genfromtxt(self.root + r'/train_test_split.txt', delimiter=' ', dtype='int')

        # 分别存放训练和测试数据集
        self.train = []  # [[img_data, img_label] ...]
        self.test = []  # [[img_data, img_label] ...]

        # 拆分训练集和测试集
        if self.is_training:
            # 11788是数据集的大小，包含训练集和测试集
            for index in range(11788):
                img_id, img_istraining = id2istraining[index]
                if img_istraining == 1:
                    self.train.append([id2data[img_id - 1][1], int(id2label[img_id - 1][1]) - 1])
        else:
            for index in range(11788):
                img_id, img_istraining = id2istraining[index]
                if img_istraining == 0:
                    self.test.append([id2data[img_id - 1][1], int(id2label[img_id - 1][1]) - 1])

    def __getitem__(self, item):
        if self.is_training:
            _data, _label = self.train[item]
            _data = cv2.imread(self.root + '/images/' + _data)
            return _data, _label
        else:
            _data, _label = self.test[item]
            _data = cv2.imread(self.root + '/images/' + _data)
            return _data, _label

    def __len__(self):
        if self.is_training:
            return len(self.train)
        else:
            return len(self.test)

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            # print(x)
            temp_path = os.path.join(self.root, x)
            # print(temp_path)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}下载数据集并且解压好")
                raise RuntimeError()


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\CUB_200_2011\CUB_200_2011\CUB_200_2011'

    # 实例化
    # 注意：返回图片为HWC格式，因为map只支持HWC；并且图片数据为uint8，即0-255，
    dataset = Cub200Generator(root=root, is_training=True)

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(dataset,
                                  shuffle=True,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  num_samples=None, )

    # 做一些数据增强，如果不需要增强可以把这段代码注释掉
    # 首先把数据集设置为uint8，因为map只支持uint8
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")
    # 此处填写所需要的数据增强算子
    transform = [cvision.Resize(448), cvision.RandomCrop(448)]
    dataset = dataset.map(operations=transform, input_columns="image")

    # 显示10张图片
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        if index >= 10:
            break
        print(data["image"].shape, data["label"])
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].squeeze(), cmap=plt.cm.gray)
        plt.title(data["label"])
    plt.show()

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


class Cub200Generator:
    def __init__(self, is_training=True, base_path=r'E:\BilinearCNN\CUB_200_2011\CUB_200_2011\CUB_200_2011'):
        self.is_training = is_training  # True表示训练集，False表示测试集
        self.base = base_path
        id2data = np.genfromtxt(self.base + r'/images.txt', delimiter=' ', dtype='str')

        id2label = np.genfromtxt(self.base + r'/image_class_labels.txt', delimiter=' ', dtype='str')

        id2istraining = np.genfromtxt(self.base + r'/train_test_split.txt', delimiter=' ', dtype='int')

        # 分别存放训练和测试数据集
        self.train = []  # [[img_data, img_label] ...]
        self.test = []  # [[img_data, img_label] ...]

        # 拆分训练集和测试集
        if self.is_training:
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
            _data = cv2.imread(self.base + '/images/' + _data).transpose((2, 0, 1)) / 255
            return _data, _label
        else:
            _data, _label = self.test[item]
            _data = cv2.imread(self.base + '/images/' + _data).transpose((2, 0, 1)) / 255
            return _data, _label

    def __len__(self):
        if self.is_training:
            return len(self.train)
        else:
            return len(self.test)

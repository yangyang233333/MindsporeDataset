# -*- coding: UTF-8 -*-
import os

import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds

import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype
import glob
import cv2
import gzip
import shutil

# 数据集下载链接
resources1 = r'https://drive.google.com/uc?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz&export=download'

# 数据集内的所有文件
resources2 = []


class WebFace:
    def __init__(self, root, is_training=True):
        """
        :param root:
        :param is_training:
        """

        self.root = root
        self.is_training = is_training  # True表示训练集，False表示测试集

        # 检查数据集是否完整
        self.check_exist()

        # 重命名文件夹，即把文件夹按照0-xxx的方式命名
        self.rename_dir()

        self.data = []
        self.label = []

        if is_training:
            self.data, self.label = self.read_data()
        else:
            self.data, self.label = self.read_data()

    def __getitem__(self, item):
        data = cv2.imread(self.data[item])
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)

    def read_data(self):
        img_list = []
        label_list = []
        index = 0
        dir_list = sorted(glob.glob(os.path.join(self.root) + r'/*'))
        for dir in dir_list:
            label = int(dir.split('\\')[-1])
            imgs = sorted(glob.glob(dir + '/*'))
            for img in imgs:
                label_list.append(label)
                img_list.append(img)
        return img_list, label_list

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}下载数据集并且解压好")
                raise RuntimeError()
        print("数据集完整！")

    def rename_dir(self):
        # 重命名文件夹，即把文件夹按照0-xxx的方式命名
        dir_list = sorted(glob.glob(os.path.join(self.root) + r'/*'))
        index = 0
        for dir in dir_list:
            if not os.path.exists(os.path.join(self.root, str(index))):
                os.rename(dir, os.path.join(self.root, str(index)))
            index += 1


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\CASIA-WebFace'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = WebFace(root=root, is_training=True)

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(dataset,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  num_samples=None,
                                  shuffle=False)

    # 做一些数据增强，如果不需要增强可以把这段代码注释掉
    # 首先把数据集设置为uint8，因为map只支持uint8
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")

    # # 此处填写所需要的数据增强算子
    # transform = [cvision.Resize(448),
    #              cvision.RandomCrop(448)]
    # dataset = dataset.map(operations=transform, input_columns="image")
    # dataset = dataset.map(operations=transform, input_columns="label")

    # 显示5组张图片
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        if index >= 5:
            break

        print(data["image"].shape, data["label"].shape)
        plt.subplot(1, 5, index + 1)
        plt.imshow(data["image"].squeeze())
        plt.title(data["label"])
    plt.show()

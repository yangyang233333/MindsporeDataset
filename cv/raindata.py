# -*- coding: UTF-8 -*-
import os

import cv2
import zipfile
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
resources1 = r'https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html'

# 数据集内的所有文件
resources2 = ["rain_data_train_Light",
              "rain_data_test_Heavy",
              "rain_data_test_Light",
              "rain_data_train_Heavy"]


class RainData:
    def __init__(self, root, is_training=True, name='Heavy'):
        """

        :param root:
        :param is_training:
        """

        self.root = root
        self.is_training = is_training  # True表示训练集，False表示测试集
        self.name = name

        # 检查数据集是否完整
        self.check_exist()

        # 解压文件
        # self.unzip()

        # 文件名预处理: 将norain-1x2.png变成norain-1.png，即移除x2
        self.remove_x2_from_filename()

        self.data = []
        self.label = []

        if is_training:
            self.data = self.read_image()
            self.label = self.read_label()
        else:
            self.data = self.read_image()
            self.label = self.read_label()

    def __getitem__(self, item):
        data = cv2.imread(self.data[item])
        label = cv2.imread(self.label[item])

        return data, label

    def __len__(self):
        return len(self.label)

    def read_image(self):
        if self.is_training:
            filename = f"rain_data_train_{self.name}"
        else:
            filename = f"rain_data_test_{self.name}"
        content = sorted(glob.glob(os.path.join(self.root, filename, 'rain') + r'\*.png'))
        return content

    def read_label(self):
        if self.is_training:
            filename = f"rain_data_train_{self.name}"
        else:
            filename = f"rain_data_test_{self.name}"
        content = sorted(glob.glob(os.path.join(self.root, filename, 'norain') + r'\*.png'))
        return content

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}下载数据集并且【解压】好")
                raise RuntimeError()
        print("数据集完整！")

    def unzip(self):
        pass
        # # 创建文件夹保存解压后的文件
        # if not os.path.exists(os.path.join(self.root, 'unzip')):
        #     os.mkdir(os.path.join(self.root, 'unzip'))
        # for item in resources2:
        #     if '.zip' in item:
        #         filename = os.path.join(self.root, 'unzip', item.split('.')[0])
        #         # print(filename)
        #     elif '.gz' in item:
        #         filename = os.path.join(self.root, 'unzip', item.split('.')[0])
        #         # print(filename)
        #         ungz_file = gzip.GzipFile(os.path.join(self.root, item))
        #         print(os.path.join(self.root, item))
        #         # print(filename)
        #         open(filename, 'wb+').write(ungz_file.read())

    def remove_x2_from_filename(self):
        # 文件名预处理: 将norain-1x2.png变成norain-1.png，即移除x2
        for item in resources2:
            dirname = os.path.join(self.root, item, 'rain')
            for img in glob.glob(dirname + '/*'):
                # print(img)
                os.rename(img, img.replace('x2', ''))


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\pku_rain_data'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = RainData(root, is_training=True, name='Light')

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(dataset,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  num_samples=None,
                                  shuffle=False)

    # 做一些数据增强，如果不需要增强可以把这段代码注释掉
    # 首先把数据集设置为uint8，因为map只支持uint8
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="label")

    # # 此处填写所需要的数据增强算子
    # transform = [cvision.Resize(448),
    #              cvision.RandomCrop(448)]
    # dataset = dataset.map(operations=transform, input_columns="image")
    # dataset = dataset.map(operations=transform, input_columns="label")

    # 显示5张图片
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        if index >= 5:
            break
        print(data["image"].shape, data["label"].shape)
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].squeeze())
        plt.title("data")

        plt.subplot(2, 5, index + 1 + 5)
        plt.imshow(data["label"].squeeze())
        plt.title("label")
    plt.show()

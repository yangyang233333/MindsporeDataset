# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype
import glob

# 数据集下载链接
resources1 = 'http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html'

# 数据集内的所有文件
resources2 = []


class Set5:

    def __init__(self, root, scale=2, is_train=False):
        """"""
        self.root = root
        self.scale = scale

        # 检查数据集是否完整
        self.check_exist()

        # 读取数据集
        if is_train:
            self.data = self.read_image()
            self.label = self.read_label()
        else:
            self.data = self.read_image()
            self.label = self.read_label()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def read_image(self):
        data = []
        if self.scale == 2:
            # x = os.path.join(self.root, r'image_SRF_2')
            # print(x)
            # print(self.root)
            # print(11111111111111111)
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_2/*'))):
                # print(img_path)
                # print(f"SRF_{self.scale}_LR.png")
                if f"SRF_{self.scale}_LR.png" in img_path:
                    # print(img_path)
                    data.append(cv2.imread(img_path)[:, :, ::-1])
        elif self.scale == 3:
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_3/*'))):
                if f"SRF_{self.scale}_LR.png" in img_path:
                    data.append(cv2.imread(img_path)[:, :, ::-1])
        elif self.scale == 4:
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_4/*'))):
                if f"SRF_{self.scale}_LR.png" in img_path:
                    data.append(cv2.imread(img_path)[:, :, ::-1])
        # print(data)
        return np.array(data)

    def read_label(self):
        label = []
        if self.scale == 2:
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_2/*'))):
                if f"SRF_{self.scale}_HR.png" in img_path:
                    label.append(cv2.imread(img_path)[:, :, ::-1])
        elif self.scale == 3:
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_3/*'))):
                if f"SRF_{self.scale}_HR.png" in img_path:
                    label.append(cv2.imread(img_path)[:, :, ::-1])
        elif self.scale == 4:
            for img_path in sorted(glob.glob(os.path.join(self.root, r'image_SRF_4/*'))):
                if f"SRF_{self.scale}_HR.png" in img_path:
                    label.append(cv2.imread(img_path)[:, :, ::-1])
        return np.array(label)

    def check_exist(self):
        pass


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\Set5'

    # 实例化
    dataset = Set5(root, scale=2, is_train=False)

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

    # 此处填写所需要的数据增强算子
    # transform = [cvision.Resize(128),
    #              cvision.RandomCrop(128)]
    # dataset = dataset.map(operations=transform, input_columns="image")
    # dataset = dataset.map(operations=transform, input_columns="label")

    # 显示图片(这个数据集只有5个图)
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        print(data["image"].shape, data["label"].shape)
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].squeeze())
        plt.title("data")

        plt.subplot(2, 5, index + 1 + 5)
        plt.imshow(data["label"].squeeze())
        plt.title("label")
    plt.show()

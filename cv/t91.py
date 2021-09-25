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
resources1 = 'https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU'

resources2 = []


class T91:

    def __init__(self, root, is_train=True, scale=2):
        """

        :param root:
        :param is_train:
        :param scale:
        """
        self.root = root

        # 检查数据集文件是否缺少
        self.check_exist()

        # 生成LR数据集 Bicubic
        print("正在生成训练集，此过程需要较长时间...")
        self.hr2lr_train_bicubic()
        print("训练集生成完毕！")

        # 生成训练集
        if is_train:
            self.label = sorted(glob.glob(os.path.join(self.root) + '\\*.bmp'))
            if scale == 2:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x2/*')))
            elif scale == 3:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x3/*')))
            elif scale == 4:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x4/*')))
        else:
            self.label = sorted(glob.glob(os.path.join(self.root) + '\\*.bmp'))
            if scale == 2:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x2/*')))
            elif scale == 3:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x3/*')))
            elif scale == 4:
                self.data = sorted(glob.glob(os.path.join(self.root, 'T91_train_LR_bicubic_x4/*')))

    def __getitem__(self, item):
        img, label = cv2.imread(self.data[item]), cv2.imread(self.label[item])
        return img, label

    def __len__(self):
        return len(self.label)

    def hr2lr_train_bicubic(self):
        """根据bicubic算法，生成train_LR数据集"""
        # 创建文件夹,用于保存生成后的数据集

        if not os.path.exists(os.path.join(self.root, r'T91_train_LR_bicubic_x2')):
            os.mkdir(os.path.join(self.root, r'T91_train_LR_bicubic_x2'))
        if not os.path.exists(os.path.join(self.root, r'T91_train_LR_bicubic_x3')):
            os.mkdir(os.path.join(self.root, r'T91_train_LR_bicubic_x3'))
        if not os.path.exists(os.path.join(self.root, r'T91_train_LR_bicubic_x4')):
            os.mkdir(os.path.join(self.root, r'T91_train_LR_bicubic_x4'))

        # 生成T91_train_LR_bicubic
        path1 = sorted(glob.glob(os.path.join(self.root) + r'/*.bmp'))
        path1_x2 = os.path.join(self.root, r'T91_train_LR_bicubic_x2')
        path1_x3 = os.path.join(self.root, r'T91_train_LR_bicubic_x3')
        path1_x4 = os.path.join(self.root, r'T91_train_LR_bicubic_x4')
        for img_path in path1:
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img_x2 = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
            img_x3 = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_CUBIC)
            img_x4 = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)

            # 保存到对应文件
            cv2.imwrite(os.path.join(path1_x2, img_path.split('\\')[-1]), img_x2)
            cv2.imwrite(os.path.join(path1_x3, img_path.split('\\')[-1]), img_x3)
            cv2.imwrite(os.path.join(path1_x4, img_path.split('\\')[-1]), img_x4)

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}网站搜索DIV2K进行下载")
                raise RuntimeError()


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\T91'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = T91(root, is_train=True, scale=2)

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

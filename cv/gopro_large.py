# -*- coding: UTF-8 -*-
import glob
import os

import cv2
import matplotlib.pyplot as plt
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ctrans

# 数据集下载链接
resources1 = r'http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large.zip'
resources11 = r'http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large_all.zip'

# 数据集内的所有文件
resources2 = []


class GoPro_Large:
    def __init__(self, root, mode='train', _input='blur'):
        """

        :param root:
        :param mode:
        :param _input:
        """

        self.root = root
        self.mode = mode  # train  test
        self.input = _input  # blur  blur_gamma

        # 检查输入参数
        if self.input not in ('blur', 'blur_gamma'):
            raise RuntimeError('参数_input输入错误，必须为blur或者blur_gamma')

        # 检查数据集是否完整
        self.check_exist()

        self.data = []
        self.label = []

        if self.mode == 'train':
            self.data, self.label = self.read_data()
        elif self.mode == 'test':
            self.data, self.label = self.read_data()
        else:
            raise NotImplementedError('{}输入有误, mode必须为train，test之一。'.format(mode))

    def __getitem__(self, item):
        data = cv2.imread(self.data[item])
        label = cv2.imread(self.label[item])

        return data, label

    def __len__(self):
        return len(self.label)

    def read_data(self):
        img_list, label_list = [], []

        list_path = os.path.join(self.root, f'{self.mode}')
        for dirs in sorted(glob.glob(list_path + '/*')):
            # print(dirs)
            img_path = os.path.join(dirs, f'{self.input}')
            for item in sorted(glob.glob(img_path + '/*')):
                img_list.append(item)

            label_path = os.path.join(dirs, f'sharp')
            for item in sorted(glob.glob(label_path + '/*')):
                label_list.append(item)
        return img_list, label_list

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                raise RuntimeError("文件{}有缺失, 请去{}或者{}下载数据集并且解压好".format(temp_path, resources1, resources11))
        print("数据集完整！")


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\GOPRO_Large'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = GoPro_Large(root=root, mode='train', _input='blur')

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

    # 显示5组张图片
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

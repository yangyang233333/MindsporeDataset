# -*- coding: UTF-8 -*-
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype

# 数据集下载链接
resources1 = r'https://github.com/rois-codh/kmnist'

# 数据集内的所有文件
resources2 = ['train-images-idx3-ubyte',
              't10k-labels-idx1-ubyte',
              't10k-images-idx3-ubyte',
              'train-labels-idx1-ubyte']


class KMNIST:

    def __init__(self, root, is_train=True):
        self.root = root

        # 检查文件是否完整
        self.check_exist()

        if is_train:
            self.data = self.read_image(os.path.join(self.root, 'train-images-idx3-ubyte'))
            self.label = self.read_label(os.path.join(self.root, 'train-labels-idx1-ubyte'))
        else:
            self.data = self.read_image(os.path.join(self.root, 't10k-images-idx3-ubyte'))
            self.label = self.read_label(os.path.join(self.root, 't10k-labels-idx1-ubyte'))

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def read_image(self, filename: str):
        """读取数据集，下面以train-images-idx3-ubyte为例介绍一下数据集结构
            数据集名字:
                file_name (string): emnist-balanced-train-images-idx3-ubyte.
            数据集的存储结构:
                file format
                [offset] [type]          [value]          [description]
                0000     32 bit integer  2051             magic number
                0004     32 bit integer  112800           number of images
                0008     32 bit integer  28               number of rows
                0012     32 bit integer  28               number of columns
                0016     unsigned byte   ??               pixel
                0017     unsigned byte   ??               pixel
                ........
                xxxx     unsigned byte   ??               pixel
        """
        file_content = open(filename, 'rb').read()
        head = struct.unpack_from(">IIII", file_content, 0)
        offset = struct.calcsize(">IIII")
        img_num = head[1]  # 图片数
        width = head[2]  # 宽度
        height = head[3]  # 高度

        bits = img_num * width * height  # data一共有img_num * width * height个像素值
        bits_string = '>' + str(bits) + 'B'  # fmt格式：'>????B'
        imgs = struct.unpack_from(bits_string, file_content, offset)  # 取data数据，返回一个元组
        imgs_array = np.array(imgs).reshape((img_num, width, height))  # 最后将读取的数据reshape成 (图片数，宽，高)的三维数组
        return imgs_array

    def read_label(self, filename):
        """ 读取数据集标签，下面以train-labels-idx1-ubyte为例介绍一下数据集结构
            数据集的名字:
                file_name (string): emnist-balanced-train-labels-idx1-ubyte.
            数据集的存储结构:
                file format
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  112800           number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
        """
        file_content = open(filename, "rb").read()
        head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
        offset = struct.calcsize('>II')
        label_num = head[1]  # label数
        bits_string = '>' + str(label_num) + 'B'  # fmt格式：'>47040000B'
        label = struct.unpack_from(bits_string, file_content, offset)  # 取data数据，返回一个元组
        return np.array(label)

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}下载数据集并且解压好")
                raise RuntimeError()


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\kmnist'

    # 实例化
    dataset = KMNIST(root=root, is_train=False)

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(dataset,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  num_samples=None,
                                  shuffle=False)

    # 做一些数据增强，如果不需要增强可以把这段代码注释掉
    # 首先把数据集设置为uint8，因为map只支持uint8
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")
    # 此处填写所需要的数据增强算子
    transform = [cvision.Resize(28), cvision.RandomCrop(28)]
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

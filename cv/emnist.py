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
resources1 = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

# 数据集内的所有文件
resources2 = ['emnist-balanced-mapping.txt',
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

    def __init__(self, root="./dataset/emnist", dataset_name="balanced", is_train=True, download=False):
        """
        :param root: 数据集文件所在的目录
        :param dataset_name: EMNIST有6个子数据集，使用此参数来指定训练/测试哪一个，balanced，By_Merge，By_Class，Digits，Letters，MNIST
        :param is_train:
        :param download:
        """

        self.root = root
        self.dataset_name = dataset_name

        # 检查数据集文件是否完整
        self.check_exist()

        # 解压所选择的子数据集
        # self.unzip()

        # 开始读取数据集
        if is_train:
            self.data = self.read_image(
                os.path.join(self.root, 'unziped', f'emnist-{self.dataset_name}-train-images-idx3-ubyte'))
            self.label = self.read_label(
                os.path.join(self.root, 'unziped', f'emnist-{self.dataset_name}-train-labels-idx1-ubyte'))
        else:
            self.data = None
            self.label = None

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def read_image(self, filename: str) -> np.ndarray:
        """读取数据集，下面以emnist-balanced-train-images-idx3-ubyte为例介绍一下数据集结构
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
        # print(imgs_array[1].shape)
        # cv2.imshow('00', imgs_array[2551]/255)
        # cv2.waitKey(0)
        return imgs_array

    def read_label(self, filename):
        """ 读取数据集标签，下面以emnist-balanced-train-labels-idx1-ubyte为例介绍一下数据集结构
            数据集的名字:
                file_name (string): emnist-balanced-train-labels-idx1-ubyte.
            数据集的存储结构:
                file format
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  60000            number of items
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
        # x = np.array(label)[:500]
        # print(np.max(x))
        return np.array(label)

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            temp_path = os.path.join(self.root, x)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/emnist.zip下载数据集并且解压好")
                raise RuntimeError()

    def unzip(self):
        """解压gz文件"""
        # 创建解压后文件的保存目录
        if not os.path.exists(self.root + "/unziped"):
            os.mkdir(self.root + "/unziped")
        file_list = glob.glob(self.root + "/*")

        for filename in file_list:
            # 将子数据集的gz文件解压到/unziped
            if self.dataset_name in filename and ".gz" in filename:
                new_file = os.path.join(self.root, "unziped", filename.split('\\')[-1].replace(".gz", ""))
                ungz_file = gzip.GzipFile(filename)  # 解压文件
                open(new_file, 'wb+').write(ungz_file.read())  # 将解压后的文件对象写入新文件

            # 将子数据集的txt文件复制到到/unziped
            elif self.dataset_name in filename and ".txt" in filename:
                new_file = os.path.join(self.root, "unziped", filename.split('\\')[-1])
                shutil.copyfile(filename, new_file)
        print("文件解压完成！")


def emnist():
    pass


if __name__ == '__main__':
    root = 'E:\MindsporeVision\dataset\emnist'
    data = _EMNIST(root=root, is_train=True)
    dataset = ds.GeneratorDataset(data, column_names=["image", "label"])

    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        if index >= 10:
            break
        print(data["image"].shape, data["label"])
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].astype(np.int8).squeeze(), cmap=plt.cm.gray)
        plt.title(data["label"])
    plt.show()

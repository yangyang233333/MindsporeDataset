import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype

# 数据集下载链接
resources1 = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]

# 数据集文件
resources2 = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]


class _FashionMNIST:
    """ Helper class for Fashion- MNIST Dataset. """

    def __init__(self, root, is_train=True):
        """

        :param root:
        :param is_train:
        """
        self.root = root

        # 检查数据集是否完整
        self.check_exist()

        # 读取训练集/测试集
        if is_train:
            self.data = self.read_image(os.path.join(self.root, "train-images-idx3-ubyte"))
            self.label = self.read_label(os.path.join(self.root, "train-labels-idx1-ubyte"))
        else:
            self.data = self.read_image(os.path.join(self.root, "t10k-images-idx3-ubyte"))
            self.label = self.read_label(os.path.join(self.root, "t10k-labels-idx1-ubyte"))

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def read_image(self, file_name):
        """ read image binary file.
            Args:
                file_name (string): file path of *-images-idx3-ubyte.
            Note:
                file format
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000803(2051) magic number
                0004     32 bit integer  60000            number of images
                0008     32 bit integer  28               number of rows
                0012     32 bit integer  28               number of columns
                0016     unsigned byte   ??               pixel
                0017     unsigned byte   ??               pixel
                ........
                xxxx     unsigned byte   ??               pixel
        """
        file_handle = open(file_name, "rb")  # 以二进制打开文档
        file_content = file_handle.read()  # 读取到缓冲区中
        head = struct.unpack_from('>IIII', file_content, 0)  # 取前4个整数，返回一个元组
        offset = struct.calcsize('>IIII')
        img_num = head[1]  # 图片数
        width = head[2]  # 宽度
        height = head[3]  # 高度

        bits = img_num * width * height  # data一共有60000*28*28个像素值
        bits_string = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
        imgs = struct.unpack_from(bits_string, file_content, offset)  # 取data数据，返回一个元组
        imgs_array = np.array(imgs).reshape((img_num, width, height))  # 最后将读取的数据reshape成 (图片数，宽，高)的三维数组
        return imgs_array

    def read_label(self, file_name):
        """ read label binary file.
            Args:
                file_name (string): file path of *-label-idx3-ubyte.
            Note:
                file format
                [offset] [type]          [value]          [description]
                0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                0004     32 bit integer  60000            number of items
                0008     unsigned byte   ??               label
                0009     unsigned byte   ??               label
                ........
                xxxx     unsigned byte   ??               label
                The labels values are 0 to 9.
        """
        file_handle = open(file_name, "rb")  # 以二进制打开文档
        file_content = file_handle.read()  # 读取到缓冲区中
        head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
        offset = struct.calcsize('>II')
        label_num = head[1]  # label数
        bits_string = '>' + str(label_num) + 'B'  # fmt格式：'>47040000B'
        label = struct.unpack_from(bits_string, file_content, offset)  # 取data数据，返回一个元组
        return np.array(label)

    def check_exist(self):
        """检查数据集文件是否完整"""
        for x in resources2:
            # print(x)
            temp_path = os.path.join(self.root, x)
            # print(temp_path)
            if not os.path.exists(temp_path):
                print(f"文件{temp_path}有缺失, 请去{resources1}下载数据集并且解压好")
                raise RuntimeError()

    # win10不支持wget，所以先注释掉再说
    # def download(self):
    #     if self.check_exist(self.root, [res.split("/")[-1] for res in resources1]):
    #         return
    #     if self.check_exist(self.root, resources2):
    #         return
    #     os.makedirs(self.root, exist_ok=True)
    #     for res in resources1:
    #         if not os.path.exists(os.path.join(self.root, res.split("/")[-1])):
    #             os.system("wget -P " + self.root + " " + res)


# 暂时不考虑这一层封装，以后再说
# def FashionMNIST(root, usage="train", sampler=None, transform=None, download=False):
#     """Fashion MNIST Dataset.
#     Args:
#         root (string): Root directory of dataset where ``*-ubyte.gz`` exist.
#         usage (string, optional): If True, creates dataset from training part,
#             otherwise from test part.
#         sampler (SamplerObj, optional): Object used to choose samples from dataset.
#         transform (callable, optional): A function/transform that takes in an image
#             and returns a transformed version. E.g, ``dataset.vision.RandomCrop``
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#     """
#     fashion_mnist = _FashionMNIST(root, usage, download)
#     dataset = ds.GeneratorDataset(fashion_mnist, column_names=["image", "label"], num_parallel_workers=4,
#                                   sampler=sampler)
#
#     if transform:
#         if not isinstance(transform, list):
#             transform = [transform]
#         # map only supports image with type uint8 now
#         dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")
#         dataset = dataset.map(operations=transform, input_columns="image")
#
#     return dataset


if __name__ == '__main__':
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\fashion-mnist'

    # 实例化
    # 注意：返回图片为HWC格式，因为map只支持HWC；并且图片数据为uint8，即0-255，
    fashion_mnist = _FashionMNIST(root, is_train=True)

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(fashion_mnist,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=False,
                                  sampler=None)

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

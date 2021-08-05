import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype

resources1 = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
              "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]

resources2 = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]


class _FashionMNIST:
    """ Helper class for Fashion- MNIST Dataset. """

    def __init__(self, root, usage="train", download=False):
        self.root = root
        if download:
            self.download()

        if self.check_exist(root, [res.split("/")[-1] for res in resources1]):
            for res in resources1:
                os.system("emnist -d " + root + "/" + res.split("/")[-1])
        elif self.check_exist(root, resources2):
            pass
        else:
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if usage == "train":
            self.data = self.read_image(os.path.join(root, "train-images-idx3-ubyte"))
            self.label = self.read_label(os.path.join(root, "train-labels-idx1-ubyte"))
        else:
            self.data = self.read_image(os.path.join(root, "t10k-images-idx3-ubyte"))
            self.label = self.read_label(os.path.join(root, "t10k-labels-idx1-ubyte"))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

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

    def check_exist(self, rt_path, resources):
        for res in resources:
            if not os.path.exists(os.path.join(rt_path, res)):
                return False
        return True

    def download(self):
        if self.check_exist(self.root, [res.split("/")[-1] for res in resources1]):
            return
        if self.check_exist(self.root, resources2):
            return
        os.makedirs(self.root, exist_ok=True)
        for res in resources1:
            if not os.path.exists(os.path.join(self.root, res.split("/")[-1])):
                os.system("wget -P " + self.root + " " + res)


def FashionMNIST(root, usage="train", sampler=None, transform=None, download=False):
    """Fashion MNIST Dataset.
    Args:
        root (string): Root directory of dataset where ``*-ubyte.gz`` exist.
        usage (string, optional): If True, creates dataset from training part,
            otherwise from test part.
        sampler (SamplerObj, optional): Object used to choose samples from dataset.
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version. E.g, ``dataset.vision.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    fashion_mnist = _FashionMNIST(root, usage, download)
    dataset = ds.GeneratorDataset(fashion_mnist, column_names=["image", "label"], num_parallel_workers=4,
                                  sampler=sampler)

    if transform:
        if not isinstance(transform, list):
            transform = [transform]
        # map only supports image with type uint8 now
        dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")
        dataset = dataset.map(operations=transform, input_columns="image")

    return dataset


if __name__ == '__main__':
    dataset_dir = "dataset/fashion-mnist"
    mnist = FashionMNIST(dataset_dir,
                         download=True,
                         sampler=ds.SequentialSampler(0, 10),
                         transform=[cvision.Resize(36), cvision.RandomCrop(28)])

    # To see what samples do we get from dataset
    for index, data in enumerate(mnist.create_dict_iterator(output_numpy=True)):
        print(data["image"].shape, data["label"])
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].squeeze(), cmap=plt.cm.gray)
        plt.title(data["label"])
    plt.show()

# EMNIST数据集文件介绍
1. 数据集基本介绍：EMNIST是对MNIST的扩展，内容主要为手写字母及数字，分为以下6类：

    + By_Class ： 共 814255 张，62 类，包含所有字母和数字，与 NIST 相比重新划分类训练集与测试机的图片数
    + By_Merge： 共 814255 张，47 类，删除了一些容易混淆的字母数字， 与 NIST 相比重新划分类训练集与测试机的图片数
    + Balanced : 共 131600 张，47 类，删除了一些容易混淆的字母数字， 每一类都包含了相同的数据，每一类训练集 2400 张，测试集 400 张
    + Digits ：共 28000 张，10 类，仅仅包含数字，每一类包含相同数量数据，每一类训练集 24000 张，测试集 4000 张
    + Letters : 共 145600 张，26 类，仅仅包含字母，每一类包含相同数据，每一类训练集5600 张，测试集 800 张
    + MNIST ： 共 70000 张，10 类，仅仅包含数字，每一类包含相同数量数据（注：这里虽然数目和分类都一样，但是图片的处理方式不一样，EMNIST 的 MNIST 子集数字占的比重更大）
    
    **其中数字0-9对应标签为0-9, 字母A-Z对应标签为10-35, 字母a-z对应标签为36-61.**
2. 使用方法
```
    """ 用法示例 """

    # 此处填写数据集的上级目录
    root = 'E:\MindsporeVision\dataset\emnist'

    # 实例化
    dataset = _EMNIST(root=root, is_train=False)

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
    transform = [cvision.Resize(36), cvision.RandomCrop(28)]
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
```
数据集的目录结构如下所示（如果目录结构和下面不一样，脚本将无法正确运行）：
```
    emnist /
        emnist - mnist - train - labels - idx1 - ubyte.gz
        emnist - mnist - train - images - idx3 - ubyte.gz
        emnist - mnist - test - labels - idx1 - ubyte.gz
        emnist - mnist - test - images - idx3 - ubyte.gz
        emnist - mnist - mapping.txt
        emnist - letters - train - labels - idx1 - ubyte.gz
        emnist - letters - train - images - idx3 - ubyte.gz
        emnist - letters - test - labels - idx1 - ubyte.gz
        emnist - letters - test - images - idx3 - ubyte.gz
        emnist - letters - mapping.txt
        emnist - digits - train - labels - idx1 - ubyte.gz
        emnist - digits - train - images - idx3 - ubyte.gz
        emnist - digits - test - labels - idx1 - ubyte.gz
        emnist - digits - test - images - idx3 - ubyte.gz
        emnist - digits - mapping.txt
        emnist - balanced - mapping.txt
        emnist - balanced - test - images - idx3 - ubyte.gz
        emnist - balanced - test - labels - idx1 - ubyte.gz
        emnist - balanced - train - images - idx3 - ubyte.gz
        emnist - byclass - test - images - idx3 - ubyte.gz
        emnist - byclass - train - images - idx3 - ubyte.gz
        emnist - byclass - train - labels - idx1 - ubyte.gz
        emnist - bymerge - test - images - idx3 - ubyte.gz
        emnist - bymerge - train - images - idx3 - ubyte.gz
        emnist - bymerge - train - labels - idx1 - ubyte.gz
        emnist - bymerge - test - labels - idx1 - ubyte.gz
        emnist - bymerge - mapping.txt
        emnist - byclass - test - labels - idx1 - ubyte.gz
        emnist - byclass - mapping.txt
        emnist - balanced - train - labels - idx1 - ubyte.gz
```
3. 数据存储大致结构如下所示：
```
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

关于此数据集的详细信息请参考此[链接](https://blog.csdn.net/Chris_zhangrx/article/details/86516331).
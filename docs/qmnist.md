# QMNIST数据集文件介绍

1. 数据集基本介绍：QMNIST是对MNIST的扩展 MNIST作为机器学习非常基础的数据集，真的不存在什么问题么？ 官方MNIST测试集仅包含10,000张随机采样图像，并且通常被认为太小而不能提供有意义的
   置信区间。这就要求从MNIST的源头来考虑这个数据集的各个方面，也就是说要求重构MNIST。 十几天前，MNIST的重构数据集QMNIST发布了。 QMNIST数据集是NIST特殊数据库中找到的
   原始数据重构生成的，并且重构了之前数据集测试集中丢失50,000张测试图像数据，形成了 完整的QMNIST数据集。

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\qmnist-master'

    # 实例化
    dataset = QMNIST(root=root, is_train=True)

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

3. 数据集的目录结构如下所示（如果目录结构和下面不一样，脚本将无法正确运行）：

```
qmnist-master/
    qmnist-test-images-idx3-ubyte.gz
    qmnist-test-labels-idx1-ubyte.gz
    ...
```

4. 数据集存储大致结构如下所示（本部分一般不需要看）：

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

关于此数据集的详细信息请参考: https://github.com/facebookresearch/qmnist



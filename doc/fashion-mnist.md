# Fashion-mnist数据集介绍

1. 数据集基本介绍：FashionMNIST 是一个替代 MNIST 手写数字集的图像数据集。
其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。FashionMNIST 的大小、
格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据
划分，28x28 的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，
且不需要改动任何的代码。
数据集预览如下所示：

![Fashion-mnist预览](img/fashion-mnist.png)

数据集文件如下所示：

| Name | Content | Examples | Size | Link | MD5 Checksum|
| --- | --- |--- | --- |--- |--- |
| `train-images-idx3-ubyte.gz`  | 训练集图片  | 60,000|26 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|`8d4fb7e6c68d591d4c3dfef9ec88bf0d`|
| `train-labels-idx1-ubyte.gz`  | 训练集标签  |60,000|29 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|`25c81989df183df01b3e8a0aad5dffbe`|
| `t10k-images-idx3-ubyte.gz`  | 测试集图片  | 10,000|4.3 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|`bef4ecab320f06d8554ea6380940ec79`|
| `t10k-labels-idx1-ubyte.gz`  | 测试集标签  | 10,000| 5.1 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|`bb300cfdad3c16e7a12a480ee83cd310`|

2. 使用方法
```
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
```

3. 数据集目录结构（如果目录结构和下面不一样，脚本将无法正确运行）
```
fashion-mnist/
    t10k-labels-idx1-ubyte
    train-labels-idx1-ubyte
    train-images-idx3-ubyte
    t10k-images-idx3-ubyte
```



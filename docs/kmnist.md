# KMNIST数据集文件介绍
1. 数据集基本介绍：KMNIST是 MNIST 数据集（28x28 灰度，70,000 张图像）的替代品，
以原始 MNIST 格式和 NumPy 格式提供。由于 MNIST 将我们限制为 10 个类，因此我们在
创建 KMNIST 时选择了一个字符来代表 10 行平假名中的每一行。数据集预览如下所示：

[KMNIST预览](img/kmnist.png)
<center>KMNIST 的 10 个类别，第一列显示每个角色的现代平假名对应物</center>

2. 使用方法
```
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
```
3. 数据集的目录结构如下所示（如果目录结构和下面不一样，脚本将无法正确运行）：
```
kmnist/
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
        
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

关于此数据集的详细信息请参考: github.com/rois-codh/kmnist
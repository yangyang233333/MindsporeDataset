# MRI 数据集文件介绍

1. 数据集基本介绍 本数据集是人类大脑的核磁共振图像（MRI），其中包括训练集、测试集、验证集。 其中每个集合都包含2类图像，即T1和T2【医学上面T1和T2分别作为临床和解刨之用】。
   ** 本脚本是将该数据集作为超分辨数据集 **

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\MRI'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    # :param root:数据集根目录
    # :param scale:放大倍数
    # :param name:train表示训练集，test表示测试集，ver表示验证集
    # :param tag:填写T1或者T2
    dataset = MRI(root=root, scale=2, name='train', tag='T1')

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

```

3. 数据集的目录结构如下所示（如果目录结构和下面不一样，脚本将无法正确运行）：

```
MRI
    test/
        T2/
        T1/
    ver/
        T2/
        T1/
    train/
        T2/
        T1/
```

4. 数据集下载链接:
   https://drive.google.com/file/d/1FbESDCxuBGGixHrF_hqqwZI238-SbFR-/view?usp=sharing
# Set5 数据集介绍

1. Set5数据集于2012年提出，常用于超分辨相关研究。该数据集一共包含5张图像，一般作为超分模型的测试指标，其中一张图像如下图所示：

![Set5_baby.png](img/set5_baby.png)

2. 使用方法
```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\Set5'

    # 实例化
    dataset = Set5(root, scale=2, is_train=False)

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

    # 此处填写所需要的数据增强算子
    # transform = [cvision.Resize(128),
    #              cvision.RandomCrop(128)]
    # dataset = dataset.map(operations=transform, input_columns="image")
    # dataset = dataset.map(operations=transform, input_columns="label")

    # 显示图片(这个数据集只有5个图)
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        print(data["image"].shape, data["label"].shape)
        plt.subplot(2, 5, index + 1)
        plt.imshow(data["image"].squeeze())
        plt.title("data")

        plt.subplot(2, 5, index + 1 + 5)
        plt.imshow(data["label"].squeeze())
        plt.title("label")
    plt.show()
```

3. 数据集目录结构（如果目录结构和下面不一样，脚本将无法正确运行）
```
Set5/
    image_SRF_4
    image_SRF_3
    image_SRF_2
```



# DIV2K 数据集介绍

1. DIV2K 数据集
DIV2K数据集是一种新发布的用于图像复原任务的高质量（2K分辨率）图像数据集。此处将该数据集用于超分辨任务。
DIV2K数据集包含800张训练图像，100张验证图像和100张测试图像。在NTIRE比赛中和Set5、Set14等一起作为基准数据集。

2. 使用方法
```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\DIV2K'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = DIV2K(root, is_train=True, scale=2)

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
    transform = [cvision.Resize(448),
                 cvision.RandomCrop(448)]
    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.map(operations=transform, input_columns="label")

    # 显示一张图片
    for index, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        if index>5:
            break
        if index != 5:
            continue

        # print(index)
        print(data["image"].shape, data["label"].shape)
        plt.subplot(2, 1, 1)
        plt.imshow(data["image"].squeeze())
        plt.title("data")

        plt.subplot(2, 1, 2)
        plt.imshow(data["label"].squeeze())
        plt.title("label")
    plt.show()
```

3. 数据集目录结构（如果目录结构和下面不一样，脚本将无法正确运行）
```
DIV2K/
    DIV2K_valid_HR
    DIV2K_train_HR
```

4. 关于该数据集的详细信息，请参考：https://data.vision.ee.ethz.ch/cvl/DIV2K/

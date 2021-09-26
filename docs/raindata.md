# PKU-RAIN-2017数据集文件介绍

1. 数据集基本介绍：PKU-RAIN-2017是北京大学于2017年发布的去雨数据集，相关工作发表于CVPR 17'。

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\pku_rain_data'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = RainData(root, is_training=True, name='Light')

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

    # 显示5张图片
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
pku_rain_data/
    rain_data_test_Heavy/
        norain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
        rain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
    rain_data_test_Light/
        norain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
        rain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
    rain_data_train_Heavy/
        norain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
        rain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
    rain_data_train_Light/
        norain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
        rain/
            norain-1.png
            norain-2.png
            norain-3.png
            ...
```

4. 关于此数据集的详细信息请参考: https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html



# Urban100 数据集介绍

1. Urban100 数据集 超分领域常用验证集，由韩国SNU大学提出。

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\Urban100'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = Urban100(root, is_training=True, scale=2)

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

3. 数据集目录结构（如果目录结构和下面不一样，脚本将无法正确运行）

```
Urban100/
    image_SRF_2/
        img_001_SRF_2_HR.png
        img_001_SRF_2_LR.png
        img_002_SRF_2_HR.png
        img_002_SRF_2_LR.png
        ...
    image_SRF_4/
        img_001_SRF_4_HR.png
        img_001_SRF_4_LR.png
        img_002_SRF_4_HR.png
        img_002_SRF_4_LR.png
        ...
```

4. 关于该数据集的详细信息，请参考：https://cv.snu.ac.kr/research/EDSR/benchmark.tar

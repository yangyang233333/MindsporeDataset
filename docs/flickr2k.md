# Flickr2K 数据集介绍

1. Flickr2K 数据集 除了DIV2K外，Flickr2K也是常用的一个数据集，它往往同DIV2K合并称作DF2K。用于进一步扩增 数据并提升模型的指标。Flickr2K包含260张2K分辨率图像，其数据存放格式与DIV2K类似.

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\Flickr2K'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = Flickr2K(root=root, is_training=True, scale=2)

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

3. 数据集目录结构（如果目录结构和下面不一样，脚本将无法正确运行）

```
Flickr2K
    Flickr2K_HR/
        000001.png
        000002.png
        000003.png
        ...
    Flickr2K_LR_bicubic/
        X2/
            000001x2.png
            000002x2.png
            000003x2.png
            ...
        X3/
            000001x3.png
            000002x3.png
            000003x3.png
            ...     
        X4/
            000001x4.png
            000002x4.png
            000003x4.png
            ...   
    Flickr2K_LR_unknown/
        X2/
            000001x2.png
            000002x2.png
            000003x2.png
            ...      
        X3/
            000001x3.png
            000002x3.png
            000003x3.png
            ...
        X4/
            000001x4.png
            000002x4.png
            000003x4.png
            ...
```

4. 关于该数据集的详细信息，请参考：http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

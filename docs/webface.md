# CASIA-WebFace数据集文件介绍

1. 数据集基本介绍：CASIA-WebFace是CASIA（中国科学院自动化研究所）李子青团队在2014年发布的针对身份鉴 定和人脸识别的数据集。原CASIA-WebFace数据集具有10575个类别494414张包含人脸的图像,
   是CASIA以一个半 自动方式从IMBb网站上收集来的人脸图像数据集。在CASIA-WebFace数据集中的每张图片由250x250个像素点构成, 每个像素点由RGB像素值表示。

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\CASIA-WebFace'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = WebFace(root=root, is_training=True)

    # 设置一些参数，如shuffle、num_parallel_workers等等
    dataset = ds.GeneratorDataset(dataset,
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  num_samples=None,
                                  shuffle=False)

    # 做一些数据增强，如果不需要增强可以把这段代码注释掉
    # 首先把数据集设置为uint8，因为map只支持uint8
    dataset = dataset.map(operations=ctrans.TypeCast(mstype.uint8), input_columns="image")

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
        plt.subplot(1, 5, index + 1)
        plt.imshow(data["image"].squeeze())
        plt.title(data["label"])
    plt.show()

```

3. 数据集的目录结构如下所示（如果目录结构和下面不一样，脚本将无法正确运行）：

```
CASIA-WebFac/
    0000045	
    0000100
    ...
```

4. 更多关于此数据集的信息可以参考： 官网地址：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html （已经被移除，无法下载）
   下载地址：https://drive.google.com/uc?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz&export=download （Google云端硬盘）

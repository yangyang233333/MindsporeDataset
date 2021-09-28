# GOPRO_大型数据集介绍

1. GOPRO_Large 数据集被提议用于动态场景去模糊。训练和测试集是公开可用的。 为了产生更真实的模糊图像，GOPRO数据集没有使用用清晰图像卷积“blur kernel”的方式，
   而是使用了一个高速摄像头快速记录下一连串清晰图像，然后将这些间隔时间非常短的 清晰图像求平均来获得模糊图像。因为相机传感器在曝光过程中是不断接收光线的，我们就
   可以理解成曝光时传感器不断接收到清晰图像的信号，合起来构成了一张模糊的图像。

2. 使用方法

```
    """ 用法示例 """

    # 填写数据集的上级目录
    root = r'E:\MindsporeVision\dataset\GOPRO_Large'

    # 实例化，注意图片为HWC、BGR格式，所以训练时候要转化为CHW、RGB格式
    dataset = GoPro_Large(root=root, mode='train', _input='blur')

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
GOPRO_Large/
    test/
        GOPR0384_11_00/
            blur/
                000001.png
                000002.png
                000003.png
                ...
            blur_gamma/
            sharp/
        ...
    train/
        GOPR0372_07_00/
            blur/
                000001.png
                000002.png
                000003.png
                ...
            blur_gamma/
            sharp/
        GOPR0372_07_01/
```

4. 关于该数据集的详细信息，请参考： （谷歌硬盘）：
   [GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
   [GOPRO_Large_all](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view?usp=sharing)
   （官网）：
   [GOPRO_Large](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large.zip)
   [GOPRO_Large_all](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large_all.zip)


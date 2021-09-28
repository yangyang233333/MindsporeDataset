# MindSpore dataset hub

This is a repo for MindSpore dataset hub

# 当前支持的数据集类别

Tips：Chrome浏览器中使用Ctrl+F可以快速查找数据集.

+ 去雨去燥去模糊
  + **GoPro Large**: [介绍](./docs/gopro_large.md)、 [代码](./cv/gopro_large.py)
  + **PKU-RAIN-2017**: [介绍](./docs/raindata.md)、 [代码](./cv/raindata.py)
  + 人脸识别
    + **LFW**: [介绍](./docs/lfw.md)、 [代码](./cv/lfw.py)
    + **CASIA-WebFace**: [介绍](./docs/webface.md)、 [代码](./cv/webface.py)
  + 图像增强、超分辨
    + **T91**: [介绍](./docs/t91.md)、 [代码](./cv/t91.py)
    + **City100**: [介绍](./docs/city100.md)、 [代码](./cv/city100.py)
    + **Flickr2K**: [介绍](./docs/flickr2k.md)、 [代码](./cv/flickr2k.py)
    + **Urban100**: [介绍](./docs/urban100.md)、 [代码](./cv/urban100.py)
    + **PIRM**: [介绍](./docs/pirm.md)、 [代码](./cv/pirm.py)
    + **Set5**: [介绍](docs/set5.md)、 [代码](./cv/set5.py)
    + **Set14**: [介绍](docs/set14.md)、 [代码](./cv/set14.py)
    + **DIV2K**: [介绍](docs/div2k.md)、 [代码](./cv/div2k.py)
    + **BSD100**: [介绍](./docs/bsd100.md)、 [代码](./cv/bsd100.py)
  + 分类、细分类、字符识别
    + **EMNIST**: [介绍](docs/emnist.md)、 [代码](./cv/emnist.py)
    + **KMNIST**: [介绍](docs/kmnist.md)、 [代码](./cv/kmnist.py)
    + **Fashion-mnist**: [介绍](docs/fashion-mnist.md)、 [代码](./cv/fashion-mnist.py)
    + **Cub200-2011**: [介绍](docs/cub200_2011.md)、 [代码](./cv/cub200_2011.py)
    + **QMNIST**: [介绍](./docs/qmnist.md)、 [代码](./cv/qmnist.py)
  + 回归
    + **CACD2000**: [介绍](./docs/cacd2000.md)、 [代码](./cv/cacd2000.py)
  + 医学图像
    + **MRI**: [介绍](./docs/mri.md)、 [代码](./cv/mri.py)

# 使用方法

找到你需要的数据集，下载/复制对应的md文件和py文件即可。

# 注意

所有的图像相关数据集加载脚本返回值格式均为HWC、BGR，且为numpy类型，所以训练时需要转为CHW和RGB格式。

# Contributor

Luoyang, Qianyangyang


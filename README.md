# MindSpore dataset hub
This is a repo for MindSpore dataset hub

# 当前支持的数据集类别
Tips：Chrome浏览器中使用Ctrl+F可以快速查找数据集.

+ CV
  + **EMNIST**: [介绍](docs/emnist.md)、 [代码](./cv/emnist.py)
  + **KMNIST**: [介绍](docs/kmnist.md)、 [代码](./cv/kmnist.py)
  + **Fashion-mnist**: [介绍](docs/fashion-mnist.md)、 [代码](./cv/fashion-mnist.py)
  + **Cub200-2011**: [介绍](docs/cub200_2011.md)、 [代码](./cv/cub200_2011.py)
  + **Set5**: [介绍](docs/set5.md)、 [代码](./cv/set5.py)
  + **Set14**: [介绍](docs/set14.md)、 [代码](./cv/set14.py)
  + **DIV2K**: [介绍](docs/div2k.md)、 [代码](./cv/div2k.py)
  + **BSD100**: [介绍](./docs/bsd100.md)、 [代码](./cv/bsd100.py)
  + **QMNIST**: [介绍](./docs/qmnist.md)、 [代码](./cv/qmnist.py)
  + **T91**: [介绍](./docs/t91.md)、 [代码](./cv/t91.py)
  + **City100**: [介绍](./docs/city100.md)、 [代码](./cv/city100.py)
  + **Urban100**: [介绍](./docs/urban100.md)、 [代码](./cv/urban100.py)
  + **PIRM**: [介绍](./docs/pirm.md)、 [代码](./cv/pirm.py)
  + **PKU-RAIN-2017**: [介绍](./docs/raindata.md)、 [代码](./cv/raindata.py)
  + **CASIA-WebFace**: [介绍](./docs/webface.md)、 [代码](./cv/webface.py)
+ to-do list
  + **Flickr2K**: [介绍](./docs/flickr2k.md)、 [代码](./cv/flickr2k.py)
  + **BSDS100**: [介绍](./docs/bsds100.md)、 [代码](./cv/bsds100.py)
  + **BSDS200**: [介绍](./docs/bsds200.md)、 [代码](./cv/bsds200.py)
  + **BSDS300**: [介绍](./docs/bsds300.md)、 [代码](./cv/bsds300.py)
  + **xxxx**: [介绍](./docs/xxx.md)、 [代码](./cv/xxx.py)
  + **xxxx**: [介绍](./docs/xxx.md)、 [代码](./cv/xxx.py)
  + **xxxx**: [介绍](./docs/xxx.md)、 [代码](./cv/xxx.py)
  + **xxxx**: [介绍](./docs/xxx.md)、 [代码](./cv/xxx.py)
  + **xxxx**: [介绍](./docs/xxx.md)、 [代码](./cv/xxx.py)

# 使用方法

找到你需要的数据集，下载/复制对应的md文件和py文件即可。

# 注意

所有的图像相关数据集加载脚本返回值格式均为HWC、BGR，且为numpy类型，所以训练时需要转为CHW和RGB格式。

# Contributor

Luoyang, Qianyangyang


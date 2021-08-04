# EMNIST数据集文件介绍
1. EMNIST是对MNIST的扩展，内容主要为手写字母及数字，分为以下6类：

    + By_Class ： 共 814255 张，62 类，包含所有字母和数字，与 NIST 相比重新划分类训练集与测试机的图片数
    + By_Merge： 共 814255 张，47 类，删除了一些容易混淆的字母数字， 与 NIST 相比重新划分类训练集与测试机的图片数
    + Balanced : 共 131600 张，47 类，删除了一些容易混淆的字母数字， 每一类都包含了相同的数据，每一类训练集 2400 张，测试集 400 张
    + Digits ：共 28000 张，10 类，仅仅包含数字，每一类包含相同数量数据，每一类训练集 24000 张，测试集 4000 张
    + Letters : 共 145600 张，26 类，仅仅包含字母，每一类包含相同数据，每一类训练集5600 张，测试集 800 张
    + MNIST ： 共 70000 张，10 类，仅仅包含数字，每一类包含相同数量数据（注：这里虽然数目和分类都一样，但是图片的处理方式不一样，EMNIST 的 MNIST 子集数字占的比重更大）
    
    **其中数字0-9对应标签为0-9, 字母A-Z对应标签为10-35, 字母a-z对应标签为36-61.**

2. 数据存储结构如下所示：
'''
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
'''


关于此数据集的详细信息请参考：https://blog.csdn.net/Chris_zhangrx/article/details/86516331
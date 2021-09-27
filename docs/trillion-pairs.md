Trillion-Pairs数据集是格林深瞳在2018年举办的万亿人脸表征比赛提供的数据集。

Trillion-Pairs数据集是格林深瞳在2018年举办的万亿人脸表征比赛提供的数据集。

Trillion-Pairs的训练集包含18万人的680万张图像，这些图像来自于数据清洗之后的MS-Celeb，加入去重之后包含10万人200万张图像的亚洲名人库数据集，去掉LFW里包含的5000个名人，最后合并训练数据。测试集包含100万人的187万张图像，来自于LFW里的名人并增加更多这些名人的其他日常照片，并加入160万干扰项，去重之后合并测试数据。同时数据集包含5个特征点的人脸关键点标签用于人脸对齐。

此数据集在BitaHub上的目录结构如下：

Trillion-Pairs                            
├─ celebrity # 亚洲名人训练集图像 ├─ celebrity_lmk # 亚洲名人关键点标签 ├─ feature_tools # 特征格式转换例程 ├─ msra # 清洗过后的MS-Celeb-1M ├─ msra_lmk
# 清洗过后的MS-Celeb-1M关键点标签 ├─ testdata # 测试集图像 └─ testdata_lmk.txt # 测试集关键点标签
更多关于此数据集的信息可以参考：http://trillionpairs.deepglint.com/overview

# -*- coding: UTF-8 -*-
import os

import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds

import mindspore.dataset.vision.c_transforms as cvision
import mindspore.dataset.transforms.c_transforms as ctrans
import mindspore.common.dtype as mstype
import glob
import cv2
import gzip
import shutil

# root = r'E:\MindsporeVision\dataset\BSD100'
# data = glob.glob(root+r"\image_SRF_2"+r"\*")
#
# hr = cv2.imread(data[0])
# lr = cv2.imread(data[1])
#
# h, w, c = hr.shape
# print(h, w, c)
#
# h, w, c = lr.shape
# print(h, w, c)


os.rename('./test2.py', './test3.py')

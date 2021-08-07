import os
import glob
import cv2
import numpy as np

# x = glob.glob(r"E:\MindsporeVision\dataset\CUB_200_2011\CUB_200_2011\CUB_200_2011/*")
# print(x)
# new_list = []
# for i in x:
#     new_list.append(i.split('\\')[-1])
# print(new_list)

# x = os.path.exists('E:\\MindsporeVision\\dataset\\CUB_200_2011\\CUB_200_2011\\CUB_200_2011\\attributes')
# print(x)

path = r'E:\MindsporeVision\dataset\CUB_200_2011\CUB_200_2011\CUB_200_2011\images\017.Cardinal\Cardinal_0055_18898.jpg'

img = cv2.imread(path)
print(img.shape)
print(np.max(img))  # 255
print(np.min(img))  # 0

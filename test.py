import os
import glob


x = glob.glob(r'E:\MindsporeVision\dataset\emnist\unziped/*')
y = os.path.join(r'E:\MindsporeVision\dataset\emnist', 'unziped', r'emnist-balanced-mapping.txt')
print(x)
print(y)
print(y in x)

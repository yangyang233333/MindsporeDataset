import gzip, os

# gz_file = r'E:\MindsporeVision\mnist\t10k-images-idx3-ubyte.gz'
#
# ungz_file = gzip.GzipFile(gz_file)
#
# print(ungz_file)

if not os.path.exists("./unziped"):
    os.mkdir("./unziped")

import numpy as np
import tensorflow as tf


def str2float(s):
    return float(s)


hello = ['1.3', '1.5', '2.8']
# hello = str(hello)
h = list(map(float, hello))
print(h)
for i in h:
    print(i)
# hello_flo = float(hello)
# a = np.mat([[1,2,1],[3,2,1]])
# print(np.histogram(a,bins=3,range=(0,3)))  #频率分布直方图

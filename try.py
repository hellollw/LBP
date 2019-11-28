import numpy as np

a = np.mat([[1,2,1],[3,2,1]])
print(np.histogram(a,bins=3,range=(0,3)))  #频率分布直方图

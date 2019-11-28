# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-27

"""
使用scikit_image实现图片的LBP特征提取(Local Binary Pattern,局部二值模式)

修改：
1. 输出的特征向量维度需要一致：
    将图片按照图片自身的大小进行分割
2. 使用sklearn去分割数据
3. 使用csv保存图片的特征向量：
    permisiion denied: 写入时打开了对应的文件

"""

import numpy as np
from skimage import io, color, filters, feature
import matplotlib.pyplot as plt
import os
import csv
import time
from sklearn import svm, multiclass, model_selection


# 获得文件夹标签索引
# 输入：图片文件夹路径:path
# 输出：文件夹标签索引:file_label_list
def getFileLabelList(path):
    file_label_list = []
    for filename in os.listdir(path):
        if filename not in file_label_list:
            file_label_list.append(filename)
        else:
            raise NameError('文件夹命名错误')
    return file_label_list


# 使用scikit_image包提取图片的LBP全局特征向量
# 输入：图片路径:path, 高斯滤波稀疏:sigma, 采样半径:radius, 图像分块数量:split
# 输出：图片的LBP全局特征向量（列表形式):lbq_vector
def getLBP_Vector(path, sigma, radius, split):
    """

    :param path: 图片路径
    :param sigma: 高斯滤波稀疏
    :param radius: 采样半径
    :param split: 图像分块数量
    :return: 图片的LBP全局特征向量（列表形式):lbq_vector
    """
    pic = io.imread(path)
    pic_gauss = filters.gaussian(pic, sigma=sigma, multichannel=True)  # 高斯滤波消除噪声，默认为多通道（RGB图像）
    roi = color.rgb2gray(pic_gauss)  # 灰度图像每一像素用0~1表示
    rows, cols = np.shape(roi)
    # 使用圆形采样
    n_points = radius * 8
    # 将一幅图片分为9小块
    split_row = split
    split_col = split
    # 初始化LBP特征向量
    lbp_vector = []
    # 对图片进行LBP特征提取
    for i in range(split_row):
        for j in range(split_col):
            # 进行图片的裁剪，最后一列需要减1！
            cur_roi = roi[rows // split_row * i:rows // split_row * (i + 1) - 1,
                      cols // split_col * j:cols // split_col * (j + 1) - 1].copy()
            # io.imshow(cur_roi)
            # plt.show()
            cur_lbpmat = feature.local_binary_pattern(cur_roi, P=n_points, R=radius, method='uniform')
            cur_lbpmat = cur_lbpmat.astype(np.int32)
            m, n = np.shape(cur_lbpmat)
            number = m * n
            max_bins = int(cur_lbpmat.max() + 1)
            cur_hist, cur_bins = np.histogram(cur_lbpmat, bins=max_bins, range=(0, max_bins))
            cur_hist = cur_hist / number  # 归一化转换为概率分布
            lbp_vector.extend(cur_hist.tolist())
    return lbp_vector


# 训练所有图像的LBP，获得训练样本和测试样本数据集和的标签集和
# 输入：样本图片文件所在路径:path, 高斯滤波稀疏:sigma, 采样半径:radius, 图像分块数量:split
# 返回：样本数据集:data_training,样本标签:label_training（后可用sklearn中的model_selection.train_test_split方法来分割数据）
def LBP_Data(path, sigma,radius,split):
    data_training = []
    label_training = []
    filelabellist = getFileLabelList(path)  #获得每个文件夹的名称，每个文件夹的名称也就对应了其所属的类别
    # i = 1   #每5个数据采集一个测试样本
    for cur_file in filelabellist:
        cur_path = path+cur_file+'/'
        for cur_jpg in os.listdir(cur_path):
            if 'jpg' in cur_jpg:
                cur_jpg_path = cur_path+cur_jpg
                cur_lbp_vector = getLBP_Vector(cur_jpg_path,sigma,radius,split)
                # if i%5==0:  #计数到5，采集一个测试样本
                #     data_test.append(cur_lbp_vector)
                #     label_test.append([cur_file,cur_jpg]) #文件夹名称同时为样本类别
                # else :  #采集训练样本
                data_training.append(cur_lbp_vector)
                label_training.append([cur_file,cur_jpg])
            else:
                continue
    csvWrite('./result/data_training.csv',data_training)
    csvWrite('./result/label_training.csv',label_training)

# 使用csv写入文件
# 输入：文件名:dataname, 列表数据:datalist
# 输出：在指定位置处写入指定姓名的文件
def csvWrite(dataname,datalist):
    f = open(dataname,'w',encoding='utf-8',newline='') #设置newline=''，不会产生空行
    csv_writer = csv.writer(f)
    for cur_data in datalist:   #datalist应为二维数组
        csv_writer.writerow(cur_data)
    f.close()
    print('写出'+dataname+'成功')

# 将标签字符串文件转换为数字文件

# 使用sklearn进行支持向量机的学习

if __name__ == '__main__':
    starttime = time.time()
    path = './temp/'
    sigma = 0.8
    radius = 3
    split = 2
    # LBP_Data(path,sigma,radius,split)
    endtime = time.time()
    dtime = endtime-starttime
    print("程序运行时间:%.8s秒" %dtime)
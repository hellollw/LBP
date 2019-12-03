# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-03

"""
使用scikit_image实现图片的LBP特征提取(Local Binary Pattern,局部二值模式)

样本总数:5640
test集比率:0.2

split: 7*7——>3234维特征向量
        3*3——>594（欠拟合）
        4*4——>1056(欠拟合）


修改：
1. 输出的特征向量维度需要一致：
    将图片按照图片自身的大小进行分割
2. 使用sklearn去分割数据
3. 使用csv保存图片的特征向量：
    permisiion denied: 写入时打开了对应的文件
4. csv读取的为字符型数据，在放入sklearn中学习前应转换为浮点型(将二维list的数据转换为浮点型）
5. 将训练得到的模板进行保存(导入pickle模块进行保存）
6. 将参数和结果也进行文件形式保存
7. 修改分类器评估指标（以普通accuracy为指标)——因为在这个多分类问题中每个类别所占权重一致
8. 数据写入时产生错误(最后会有数据没有写入？），使列表的维度不一致，np.shape()返回不了值——可以使用try except写，防止错误产生
9. 数据写入操作时会产生遗漏（自动默认填补）

"""

import numpy as np
from skimage import io, color, filters, feature
import os
import csv
import time
from sklearn import svm, multiclass, model_selection
import joblib as job


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
# 输入：图片路径:path, 高斯滤波稀疏:sigma, 采样半径:radius, 图像分块数量:split(split*split)
# 输出：图片的LBP全局特征向量（列表形式):lbq_vector，特征数量
def getLBP_Vector(path, sigma, radius, split):
    """

    :param path: 图片路径
    :param sigma: 高斯滤波稀疏
    :param radius: 采样半径
    :param split: 图像分块数量
    :return: 图片的LBP全局特征向量（列表形式):lbq_vector,图片的特征数量
    """
    pic = io.imread(path)
    pic_gauss = filters.gaussian(pic, sigma=sigma, multichannel=True)  # 高斯滤波消除噪声，默认为多通道（RGB图像）
    roi = color.rgb2gray(pic_gauss)  # 灰度图像每一像素用0~1表示
    rows, cols = np.shape(roi)
    # 使用圆形采样
    n_points = radius * 8
    # 将一幅图片分为cell小块
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
    return lbp_vector, np.shape(lbp_vector)[0]


# 训练所有图像的LBP，获得训练样本和测试样本数据集和的标签集和,并将其保存在result中
# 输入：样本图片文件所在路径:path, 高斯滤波稀疏:sigma, 采样半径:radius, 图像分块数量:split
# 返回：样本数据集:data_training,样本标签:label_training（后可用sklearn中的model_selection.train_test_split方法来分割数据）
def trainLBP_Data(path, sigma, radius, split):
    """

    :param path: 样本图片文件所在路径
    :param sigma: 高斯滤波稀疏
    :param radius: 采样半径
    :param split: 图像分块数
    :return: 样本数据集:data_training,样本标签:label_training
    """
    data_training = []
    label_training = []
    character_num = 0
    filelabellist = getFileLabelList(path)  # 获得每个文件夹的名称，每个文件夹的名称也就对应了其所属的类别
    # i = 1   #每5个数据采集一个测试样本
    for cur_file in filelabellist:
        cur_path = path + cur_file + '/'
        for cur_jpg in os.listdir(cur_path):
            if 'jpg' in cur_jpg:
                cur_jpg_path = cur_path + cur_jpg
                cur_lbp_vector, character_num = getLBP_Vector(cur_jpg_path, sigma, radius, split)
                # if i%5==0:  #计数到5，采集一个测试样本
                #     data_test.append(cur_lbp_vector)
                #     label_test.append([cur_file,cur_jpg]) #文件夹名称同时为样本类别
                # else :  #采集训练样本
                data_training.append(cur_lbp_vector)
                label_training.append([cur_file, cur_jpg])
            else:
                continue
        print('文件夹' + cur_file + '处理完毕')
    csvWrite('./result/data_training' + str(split) + '.csv', data_training)
    csvWrite('./result/label_training' + str(split) + '.csv', label_training)
    f = open('./result/LBPparameters.txt', 'a+')  # 连续写入
    f.write('sigma:%f radius:%f split:%f character_num:%d' % (sigma, radius, split, character_num) + '\r\n')
    f.close()


# 使用csv写入文件
# 输入：文件名:dataname, 列表数据:datalist
# 输出：在指定位置处写入指定姓名的文件
def csvWrite(dataname, datalist):
    f = open(dataname, 'w', encoding='utf-8', newline='')  # 设置newline=''，不会产生空行
    csv_writer = csv.writer(f)
    for cur_data in datalist:  # datalist应为二维数组
        csv_writer.writerow(cur_data)
    f.close()
    print('写出' + dataname + '成功')


# 将标签字符串文件转换为数字文件(符合sklearn的要求）
# 输入：字符串样本集和labelstring
# 输出：转换为对应的数字labelint, 集和种类：labelnum
def string2int(labelstring):
    print(np.shape(labelstring))
    Wholelabel = []
    labelint = []
    for label in labelstring:
        if label not in Wholelabel:
            Wholelabel.append(label)
        else:
            continue

    for cur_label in labelstring:
        label_index = Wholelabel.index(cur_label)
        labelint.append(label_index)  # 转换为对应的种类数字

    return labelint, len(Wholelabel)


# 将二维字符串型数组转换为浮点型
# 输入：二维字符串数组:datastr，检测维度大时候shape为什么会出错:split
# 输出：浮点数组:datafloat
def str2float(datastr, split):
    m = 5640
    if split == 7:  # 针对7的特殊情况
        n = 3234
    elif split == 6:
        n = 2376
    elif split == 5:
        n = 1650
    elif split == 4:
        n = 1056
    datafloat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            try:
                datafloat[i][j] = float(datastr[i][j].rstrip())  # 数组遍历实现转换
            except:
                print('为0位置在于%d行%d列' % (i + 1, j + 1))
                datafloat[i][j] = 0.0
    return datafloat


# 使用sklearn进行支持向量机的学习和测试
# 输入：数据集路径:path,内核选取：kernal, 惩罚系数:C，核函数系数:gamma,读取的文件名字:split
# 输出：错误率:wrongrate,分类数量:class_num
def sklearnPLB(path, kernal, C, gamma, split):
    """
    占时免去保存作用
    先读取CSV文件,之后使用sklearn进行数据分类
    :param path: 数据集路径
    :param kernal: 内核选取
    :param C: 惩罚系数
    :param gamma: 核函数系数
    :return: 错误率:wrongrate,分类数量:class_num
    """
    sampledatastr = []
    samplelabelstr = []
    # 读取训练数据集数据
    f = open(path + 'data_training' + str(split) + '.csv', 'r')
    csv_read = csv.reader(f)
    # numi = 0
    for i in csv_read:
        sampledatastr.append(i)
        # numi+=1
        # if numi>300:   #测试是否是因为列表过大
        #     break
    f.close()
    # 读取训练数据集样本数据
    f2 = open(path + 'label_training' + str(split) + '.csv', 'r')
    csv_read2 = csv.reader(f2)
    # numj = 0
    for j in csv_read2:
        samplelabelstr.append(j[0])  # 选择第一项写入样本序列中
        # numj+=1
        # if numj>300:
        #     break
    f2.close()
    # 将样本集转换为数字,将数据集转换为浮点型
    samplelabel, class_num = string2int(samplelabelstr)
    sampledataf = str2float(sampledatastr, split)
    print('数据读取完成')
    # 样本集分割
    X_train, X_test, y_train, y_test = model_selection.train_test_split(sampledataf, samplelabel, test_size=0.2,
                                                                        random_state=0)  # 划分训练集和测试集，将输入的列表分离(
    # 选取1/5的数据作为测试集)
    print('训练集分类完成')
    # 构建训练内核
    svc_rbf = svm.SVC(C=C, kernel=kernal, gamma=gamma)
    # 构建多分类器
    model = multiclass.OneVsOneClassifier(svc_rbf, -1)  # n-jobs, -1代表利用所有的cpu资源
    # 进行训练
    print('进入训练')
    clf = model.fit(X_train, y_train)
    # job.dump(clf,'./model/train_model.m')   #将模板进行保存
    test_accuracy = clf.score(X_test, y_test)
    train_accuracy = clf.score(X_train, y_train)
    print(test_accuracy)
    print(train_accuracy)
    return test_accuracy, train_accuracy, class_num


if __name__ == '__main__':
    starttime = time.time()
    # 定义LBP特征参数
    path = 'D:/MachineLearning_DataSet/DTD_DescrubableTextures/dtd/images/'  # 训练集路径
    # path = './temp/'
    sigma = 0.8
    radius = 8
    for split in range(4, 8):
        # split = 7
        # 定义SVM特征参数
        C = 10
        gamma = 0.8  # RBF系数
        # trainLBP_Data(path, sigma, radius, split)
        testaccuracy, trainaccuracy, class_num = sklearnPLB('./result/', kernal='rbf', C=C, gamma=gamma, split=split)
        endtime = time.time()
        dtime = endtime - starttime
        f = open('./result/SVMparameters3.txt', 'a+')  # 连续写入
        f.write('训练样本集为:' + str(split) + '\n')
        f.write('C:%.2f gamma:%.1f' % (C, gamma) + '\n')
        f.write('Result: testAccuracy:%.4f, trainAccuracy:%.4f,class_num:%d, RunTime:%.8s' % (testaccuracy,
                                                                                              trainaccuracy,
                                                                                              class_num,
                                                                                              dtime) + '\r\n')
        f.close()
        print("程序运行时间:%.8s秒" % dtime)
        print('第%d次读取完成' % (split - 3))

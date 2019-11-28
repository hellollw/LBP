


########################################################################################################





import numpy as np
import cv2
import os
from skimage import io, transform, color, measure, segmentation, morphology, feature
from sklearn import svm, multiclass, model_selection
import csv  #csv文件
import matplotlib.pyplot as plt
# from PIL import Image
 
path = "G:\\Sample1\\"
csvfile = path+"ground_truth.csv"
print(csvfile)
pic_path = []
label_1 = []
label_2 = []
 
 
#把标签转换为 数字量
def label2number(label_list):
    label=np.zeros(len(label_list),)
    label_unique=np.unique(label_list)
    num=label_unique.shape[0]
    # label_list = np.array(label_list)
    for k in range(num):
        temp=label_unique[k]
        index=[i for i, v in enumerate(label_list) if v == temp]
        #
        # print(temp)
        # index=label_list.find(temp)
        label[index]=k
    return label,label_unique
 
# 填充空白区域
def imfill(im_th):
    # im_th 是0 1 整形二值图
    # Copy the thresholded imapge.
    im_th = np.uint8(im_th)
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    # print('h '+str(h)+'   w '+str(w))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 1)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out
 
 
#打开csv文件  第0列是 图片名称  第1 2列是 两种标签
with open(csvfile, "r") as f:
    # with open(birth_weight_file, "w") as f:
    csvreader = csv.reader(f)
    csvheader = next(csvreader)
    print(csvheader)
    for row in csvreader:
 
        # print(len(row))
        pic_path.append(path+'Images\\'+row[0])
        label_1.append(row[1])
        label_2.append(row[2])
 
 
# 图片样本进行预处理，进行裁剪，去除非必要部分
vidHeight = 660
vidWidth = 1120
# for i in range(0, len(pic_path)):
Data=[]
for i in range(0, 1000):
    if os.path.exists(pic_path[i]):
        pic_temp = io.imread(pic_path[i])   #读取图片
 
        pic_temp = pic_temp[300:(660+300), 80:(80+1120)]    #进行图像裁剪

        roi = color.rgb2gray(pic_temp)

        thresh = 140
        bw = (roi <= thresh/255) * 1  # 根据阈值进行分割，将灰度图转换为二值图像（
        # dst=np.uint8(dst)
        pic_temp2 = imfill(bw)  #填充？
 
        cleared = pic_temp2.copy()  # 复制
        segmentation.clear_border(cleared)  # 清除与边界相连的目标物
        label_image = measure.label(cleared)  # 连通区域标记  connectivity=1 # 4连通区域标记
        # image_label_overlay = color.label2rgb(label_image)  # 不同标记用不同颜色显示
        # plt.imshow(image_label_overlay, interpolation='nearest')
        # plt.show()
        borders = np.logical_xor(bw, cleared)  # 异或,去除背景
        label_image[borders] = -1
        Eccentricity = 1   # 离心率
        for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
            # 忽略小区域
 
            if region.area < 100000:
                continue
            # print('area is ' + str(region.area) + '  ecc is' + str(region.eccentricity))
            if Eccentricity > region.eccentricity:
                Eccentricity = region.eccentricity
                minr, minc, maxr, maxc = region.bbox  # 绘制外包矩形
        # 判断是否有符合条件的区域
        if 'minr' in vars():
            pic = pic_temp[minr:maxr, minc:maxc,:]
            pic = transform.resize(pic, [256,256,3])
            #print(lbp)
            # plt.imshow(pic)
            # plt.show()
        else:
            pic = transform.resize(pic_temp, [256, 256, 3])
 
        #提取LBP特征，每个图像分成4块进行提取
        pic1 = color.rgb2gray(pic)
        rows, cols = pic1.shape
        radius = 2;
        n_points = radius * 8
 
        lbp_sum=[]
        for row in range(2):
            for col in range(2):
                #print(str((row * rows//2)) + ' : ' + str(((row+1) * rows//2 - 1)))
                pic1_block = pic1[(row * rows//2) : ((row+1) * rows//2 - 1) , (col * col//2) : ((col+1) * col//2 - 1)]
                lbp = feature.local_binary_pattern(pic1, n_points, radius, 'uniform')
                lbp2 = lbp.astype(np.int32)
                max_bins = int(lbp2.max() + 1)
                train_hist, _ = np.histogram(lbp2, normed=True, bins=max_bins, range=(0, max_bins))
               # print(train_hist.dtype)
                #print(train_hist)
                lbp_sum=lbp_sum + train_hist.tolist()
                #
 
        Data.append(lbp_sum)
# 使用SVM进行训练并计算测试准确率

label1, _ = label2number(label_1[0:1000])
#制作训练集和样本集
X_train,X_test, y_train, y_test = model_selection.train_test_split(Data,label1,test_size=0.2, random_state=0) #程序帮忙划分训练集和测试集，将输入的列表分离

svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1);  #选用高斯内核，这里进行SVM回归训练，gamma为核函数系数，C为惩罚系数
model = multiclass .OneVsRestClassifier(svr_rbf,-1)  #n-jobs, -1代表利用所有的cpu资源
clf = model.fit(X_train, y_train)
sore=clf.score(X_test, y_test)  #获得训练误差
print('acc'+str(sore))

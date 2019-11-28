# LBP
Using scikit_image and sklearn implement texture recognition

使用scikit_image：
    1. 对图片进行高斯滤波
    2. 对图片进行LBP特征提取，获得全局LBP特征向量进行分类

使用scikit_learn:
    1. 对数据样本进行分类
    2. 使用svm的svc中的rbf内核
    3. 使用multiclass进行多分类

这些库真香！

文件分布：
    model保存训练的模型数据
    result保存LBP特征分类的结果
    temp保存用来测试的例程
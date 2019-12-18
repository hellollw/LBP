from skimage import io
import pathlib
import numpy as np

#创建文件名队列
path = 'D:\MachineLearning_DataSet\DTD_DescrubableTextures\dtd\images'
path = pathlib.Path(path)
image_path_list = list(path.glob('*/*'))
image_path_list = [str(cur_path) for cur_path in image_path_list]
min_weight = np.inf
min_height = np.inf
for cur_image_path in image_path_list:
    if 'jpg' not in cur_image_path:
        print(cur_image_path)
    else:
        image = io.imread(cur_image_path)
        image_height,image_weight,dim = np.shape(image)
        if image_height<min_height:
            min_height = image_height
        if image_weight<min_weight:
            min_weight = image_weight
print(min_height)
print(min_weight)
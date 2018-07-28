
import cv2
import gdal
import numpy as np
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt


img_path = '../../data/originaldata/NatureProtected/src/sample1.png'

out_dir='../../data/'

window_size=2048

if __name__=='__main__':
    img = cv2.imread(img_path)

    print(img.shape)

    dataset = gdal.Open(img_path)
    if dataset==None:
        print("open failed!\n")

    height = dataset.RasterYSize
    width=dataset.RasterXSize
    bands = dataset.RasterCount
    all_data=dataset.ReadAsArray(0,0,width,height)
    # all_data = np.array(all_data)
    # new_data = img_to_array(all_data)

    # print("shape:{}".format(new_data.shape))

    x = np.random.randint(0, height - window_size - 1)
    y = np.random.randint(0, width - window_size - 1)

    output_img = all_data[:, x:x + window_size, y:y + window_size]
    print(output_img.shape)
    result = output_img[1:4,:,:]
    result = np.transpose(result,(1,2,0))
    plt.imshow(result)
    plt.show()






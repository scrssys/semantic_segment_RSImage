#coding:utf-8

from PIL import Image
import cv2
import numpy as np
from base_functions import load_img_by_gdal
import matplotlib.pyplot as plt
import gdal
import sys


input_src_file = '/home/omnisky/PycharmProjects/data/originaldata/4bands/test/LZ_GF2_Clip10.png'

clip_src_file = '/home/omnisky/PycharmProjects/data/test/shuidao/GF2shuitian22_test_4bands10.png'

window_size = 2048

if __name__=='__main__':
    # img = load_img_by_gdal(input_src_file)
    dataset = gdal.Open(input_src_file)
    if dataset==None:
        print("Open file failed:{}".format(input_src_file))
        sys.exit(-1)
    # assert (ret==0)
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    d_type = dataset.GetRasterBand(1).DataType
    img = dataset.ReadAsArray(0,0,width,height)
    del dataset

    x = np.random.randint(0, height-window_size-1)
    y = np.random.randint(0, width - window_size - 1)

    if im_bands ==1:
        output_img = img[x:x + window_size, y:y + window_size]
        # output_img = img[0:24000, 0:12500]
        output_img = np.array(output_img, np.uint16)
        output_img[output_img > 1] = 127
        print(np.unique(output_img))
        output_img = np.array(output_img, np.uint8)
        plt.imshow(output_img)
        plt.show()
        cv2.imwrite(clip_src_file, output_img)  # for label clip
    else:
        output_img = img[:, x:x + window_size, y:y + window_size]
        # output_img = img[:, 0:24000, 0:12500]
        plt.imshow(output_img[0])
        plt.show()
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(clip_src_file, window_size, window_size, im_bands, d_type)
        # outdataset = driver.Create(clip_src_file, 12500, 24000, im_bands, d_type)
        if outdataset == None:
            print("create dataset failed!\n")
            sys.exit(-2)
        if im_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(output_img)
        else:
            for i in range(im_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(output_img[i])
        del outdataset









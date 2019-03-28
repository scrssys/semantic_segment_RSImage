import os
import sys

import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2


input_file = '/home/omnisky/PycharmProjects/data/test/tianfuxinqu/ecognition/original/c2017_GJ_Clip_SVM.v1.tif'
output_file = '/home/omnisky/PycharmProjects/data/test/tianfuxinqu/ecognition/pred/2017_GJ_Clip_SVM.png'
if __name__=='__main__':
    if not os.path.isfile(input_file):
        print("Not valid file path or name:{}".format(input_file))
        sys.exit(-1)

    dataset_in = gdal.Open(input_file)
    if None==dataset_in:
        print("Open file failed!")
        sys.exit(-2)
    width = dataset_in.RasterXSize
    height = dataset_in.RasterYSize
    im_bands = dataset_in.RasterCount
    # im_type = dataset_in.

    img= dataset_in.ReadAsArray(0,0,width,height)
    mask = np.zeros((height,width,), np.uint8)

    # for b in im_bands:
    #     tmp = img[b,:,:]
    #     indx = np.where(tmp)
    bk_data = img[2, :, :]
    # plt.imshow(bk_data, cmap='gray')
    # plt.show()


    roads_data = img[0,:,:]
    indx = np.where(roads_data == 255)
    mask[indx] =1
    # plt.imshow(roads_data)
    # plt.show()

    buildings_data = img[1, :, :]
    indx = np.where(buildings_data == 255)
    mask[indx] = 2
    # plt.imshow(buildings_data)
    # plt.show()
    plt.imshow(mask)
    plt.show()

    cv2.imwrite(output_file, mask)




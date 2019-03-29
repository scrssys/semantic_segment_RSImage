import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import sys
import gdal
from keras.preprocessing.image import img_to_array
from scipy.signal import medfilt, medfilt2d
from skimage import exposure


filename = '/home/omnisky/PycharmProjects/data/traindata/APsamples/binary/buildings/3.png'

outputfile = '/home/omnisky/PycharmProjects/data/traindata/APsamples/binary/buildings/3_medfit5.png'

def rotate(xb, angle):
    xb = np.rot90(np.array(xb), k=angle)
    # yb = np.rot90(np.array(yb), k=angle)
    return xb

def add_noise(xb):
    for i in range(1000):
        temp_x = np.random.randint(0, xb.shape[1])
        temp_y = np.random.randint(0, xb.shape[2])
        xb[:, temp_x, temp_y] =255
    return xb



def data_augment(xb):
    # xb = exposure.adjust_gamma(xb, 1.0)

    # xb = np.transpose(xb,(1,2,0))
    # xb = rotate(xb, 1)

    # xb = rotate(xb, 2)
    #
    # xb = rotate(xb, 3)
    # xb = np.transpose(xb, (2, 0, 1))
    #
    # xb = np.transpose(xb, (1, 2, 0))
    # xb = np.fliplr(xb)  # flip an array horizontally
    # xb = np.transpose(xb, (2, 0, 1))
    #
    # xb = np.transpose(xb, (1, 2, 0))
    # xb = np.flipud(xb)  # flip an array vertically (up down directory)
    # xb = np.transpose(xb, (2, 0, 1))
    #
    # xb = exposure.adjust_gamma(xb, 2.0)
    #

    xb = np.transpose(xb, (1, 2, 0))
    for i in range(3):
        xb[:,:,i] = medfilt(xb[:,:,i], (5, 5))
    xb = np.transpose(xb, (2, 0, 1))
    #
    # xb = add_noise(xb)


    return xb



if __name__=='__main__':
    print("[INFO] open file")

    dataset = gdal.Open(filename)
    if dataset == None:
        print("open failed!\n")

    Y_height = dataset.RasterYSize
    X_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    data_type = dataset.GetRasterBand(1).DataType

    src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
    src_img = np.array(src_img)
    del dataset
    # src_img = np.transpose(src_img, (2, 1, 0))

    print("[INFO] augmentation ")

    src_img = data_augment(src_img)

    # src_img = np.transpose(src_img, (1, 2, 0))

    driver = gdal.GetDriverByName("GTiff")
    # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
    outdataset = driver.Create(outputfile, X_width,Y_height, im_bands, data_type)
    if im_bands == 1:
        outdataset.GetRasterBand(1).WriteArray(src_img)
    else:
        for i in range(im_bands):
            outdataset.GetRasterBand(i + 1).WriteArray(src_img[i])
    del outdataset


# coding=utf-8

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

from ulitities.base_functions import get_file, load_img_by_gdal

# seed = 1
# np.random.seed(seed)

img_w = 256
img_h = 256

# valid_labels=[0,1,2] # ignore Nodata

valid_labels=[0,1]
target_label = 1 # used for binary: 1: roads or shuidao,huapo,tuitiantu,rice; 2: buildings

# FLAG_BINARY = False  # for multiclass
FLAG_BINARY = True  #for binary


# input_path = '../../data/originaldata/zs/'
# input_path = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/originaldata/ssj/'
input_path = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/originalLabelandImages/rice/'

# output_path = '../../data/traindata/sat_urban_nrg/multiclass/'
# output_path = '../../data/traindata/test_3/multiclass'
# output_path = '../../data/traindata/huapo_512/'
# output_path = '../../data/traindata/sat_4bands_224/binary/buildings/'
output_path = '../../data/traindata/rice/'

def rotate(xb, yb, angle):
    a,b,c=xb.shape
    if a <c:
        xb = xb.transpose(1,2,0)
    xb = np.rot90(np.array(xb), k=angle)
    if a<c:
        xb = xb.transpose(2,0,1)

    yb = np.rot90(np.array(yb), k=angle)

    return xb, yb

def add_noise(xb,dtype=1):
    if dtype==1:
        noise_value=255
    elif dtype ==2:
        noise_value =1024
    else:
        noise_value = 65535
    a, b, c = xb.shape
    if a > c:
        xb = xb.transpose(2,0,1)
    tmp = np.random.random()/20.0  # max = 0.05
    noise_num =int(tmp*img_w*img_h)
    for i in range(noise_num):
        temp_x = np.random.randint(0, xb.shape[1])
        temp_y = np.random.randint(0, xb.shape[2])
        xb[:, temp_x, temp_y] = noise_value
    # if a > c:
    #     xb = xb.transpose(2,0,1)
    return xb


def data_augment(xb, yb, d_type=1):
    if np.random.random() < 0.25:
        assert(yb.shape[0]==yb.shape[1])
        assert (xb.shape[1] == xb.shape[2])
        xb, yb = rotate(xb, yb, 1)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 2)
    if np.random.random() < 0.25:
        assert (yb.shape[0] == yb.shape[1])
        assert (xb.shape[1] == xb.shape[2])
        xb, yb = rotate(xb, yb, 3)
    if np.random.random() < 0.25:
        a, b, c = xb.shape
        if a < c:
            xb = xb.transpose(1,2,0)
        # xb = np.transpose(xb, (1, 2, 0))
        xb = np.fliplr(xb)  # flip an array horizontally
        if a < c:
            xb = xb.transpose(2, 0, 1)
        yb = np.fliplr(yb)
    if np.random.random() < 0.25:
        a, b, c = xb.shape
        if a < c:
            xb = xb.transpose(1, 2, 0)
        xb = np.flipud(xb)  # flip an array vertically (up down directory)
        if a < c:
            xb = xb.transpose(2, 0, 1)
        yb = np.flipud(yb)

    if np.random.random() < 0.25:  # gamma adjust
        tmp = np.random.random()*3  #max = 3.0
        if tmp < 0.6:
            tmp = 0.6
        if tmp > 2.0:
            tmp = 2
        a, b, c = xb.shape
        if a > c:
            xb = xb.transpose(2, 0, 1)
        xb = exposure.adjust_gamma(xb, tmp)

    if np.random.random() < 0.25:  # medium filtering
        xb = xb.astype(np.float32)
        a, b, c = xb.shape
        if a < c:
            xb = xb.transpose(1, 2, 0)
        # xb = np.transpose(xb, (1, 2, 0))
        _, _,bands = xb.shape
        for i in range(bands):
            xb[:,:,i] = medfilt2d(xb[:,:,i],(3,3))
        if a<c:
            xb = np.transpose(xb, (2, 0, 1))
        xb = xb.astype(np.uint16)

    if np.random.random() < 0.2:
        xb = add_noise(xb, d_type)

    return xb, yb


def produce_training_samples_binary(in_path, out_path, image_num=50000, mode='original'):
    print('\ncreating dataset...')

    label_files,tt = get_file(os.path.join(in_path,'label/'))
    assert(tt!=0)

    image_each = image_num/len(label_files)

    print("\n[INFO] produce samples---------------------")
    g_count = 0
    for label_file in tqdm(label_files):

        src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
        if not os.path.isfile(src_file):
            print("Have no file:".format(src_file))
            continue

        print("src file:{}".format(os.path.split(src_file)[1]))

        # label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        label_img = load_img_by_gdal(label_file, grayscale=True)
        # print("label_img: {}".format(np.unique(label_img)))
        label_img = label_img.astype(np.uint8)
        y,x = label_img.shape
        # print("label_img: {}".format(np.unique(label_img)))

        dataset = gdal.Open(src_file)
        if dataset == None:
            print("open failed!\n")
            continue

        Y_height = dataset.RasterYSize
        X_width = dataset.RasterXSize

        # check size of label and src images
        x, y = label_img.shape
        if(X_width!=x and Y_height!=y):
            print("label and source image have different size:".format(label_file))
            continue

        im_bands = dataset.RasterCount
        data_type = dataset.GetRasterBand(1).DataType

        src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
        src_img = np.array(src_img)

        del dataset

        index = np.where(label_img == target_label)
        all_label = np.zeros((Y_height, X_width), np.uint8)
        all_label[index] = 1

        print(np.unique(all_label))
        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, Y_height - img_h - 1)
            src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = all_label[random_height: random_height + img_h, random_width: random_width + img_w]


            """ignore nodata area"""
            FLAG_HAS_NODATA = False
            tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
            for tt in tmp:
                if tt not in valid_labels:
                    FLAG_HAS_NODATA = True
                    continue

            if FLAG_HAS_NODATA == True:
                continue

            """ignore pure background area"""
            if len(np.unique(label_roi)) < 2:
                if 0 in np.unique(label_roi):
                    continue

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi, data_type)

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/label/%d.png' % g_count), label_roi)

            src_sample_file = out_path + '/src/%d.png' % g_count
            driver = gdal.GetDriverByName("GTiff")
            # driver = gdal.GetDriverByName("PNG")
            # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
            outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
            if outdataset == None:
                print("create dataset failed!\n")
                sys.exit(-2)
            if im_bands ==1:
                outdataset.GetRasterBand(1).WriteArray(src_roi)
            else:
                for i in range(im_bands):
                    outdataset.GetRasterBand(i+1).WriteArray(src_roi[i])
            del outdataset


            count += 1
            g_count += 1

class SelfDefinedExceptions(Exception):
    def __init__(self):
        pass

    def __str__(self):
        print("Can not find the position in label\n")
        print("The label may have different size from src image")


def produce_training_samples_multiclass(in_path, out_path, image_num=50000, mode='original'):
    print('\ncreating dataset...')

    label_files,tt = get_file(os.path.join(in_path,'label/'))
    assert(tt!=0)

    image_each = image_num/len(label_files)

    g_count = 0
    for label_file in tqdm(label_files):

        src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
        if not os.path.isfile(src_file):
            print("Have no file:".format(src_file))
            continue
            # sys.exit(-1)

        print("src file:{}".format(os.path.split(src_file)[1]))

        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        dataset = gdal.Open(src_file)
        if dataset == None:
            print("open failed!\n")
            continue

        X_height = dataset.RasterYSize
        X_width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        data_type = dataset.GetRasterBand(1).DataType

        # check size of label and src images
        x,y= label_img.shape
        print("Heigh, width of label is :{}, {}".format(x, y))
        print("Heigh, width of src is :{}, {}".format(X_height, X_width))
        if x!=X_height or y!=X_width:
            print("Warning: src and label have different size!")
            continue


        src_img = dataset.ReadAsArray(0, 0, X_width, X_height)
        src_img = np.array(src_img)

        del dataset

        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            # try:
            #     label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            # except SelfDefinedExceptions as e_result:
            #     print(e_result)

            """ignore nodata area"""
            FLAG_HAS_NODATA = False
            tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
            for tt in tmp:
                if tt not in valid_labels:
                    FLAG_HAS_NODATA = True
                    continue

            if FLAG_HAS_NODATA == True:
                continue

            """ignore pure background area"""
            if len(np.unique(label_roi)) < 2:
                if 0 in np.unique(label_roi):
                    continue
            # print(np.unique(label_roi))

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi, data_type)

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/label/%d.png' % g_count), label_roi)

            src_sample_file = out_path + '/src/%d.png' % g_count
            driver = gdal.GetDriverByName("GTiff")
            outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
            if outdataset == None:
                print("create dataset failed!\n")
                sys.exit(-2)
            if im_bands ==1:
                outdataset.GetRasterBand(1).WriteArray(src_roi)
            else:
                for i in range(im_bands):
                    outdataset.GetRasterBand(i+1).WriteArray(src_roi[i])
            del outdataset

            count += 1
            g_count += 1


if __name__ == '__main__':

    """check input directories"""
    if not os.path.isdir(os.path.join(input_path,'src/')):
        print("No input src directory:{}".format(os.path.join(input_path,'src/')))
        sys.exit(-1)
    if not os.path.isdir(os.path.join(input_path,'label/')):
        print("No input label directory:{}".format(os.path.join(input_path,'label/')))
        sys.exit(-2)

    if FLAG_BINARY==True:
        print("Produce labels for binary classification")
        produce_training_samples_binary(input_path, output_path, 300000, mode='augment')
    else:
        print("produce labels for multiclass")
        produce_training_samples_multiclass(input_path, output_path, 500, mode='augment')







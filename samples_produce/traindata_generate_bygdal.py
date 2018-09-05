# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import sys
import gdal
from keras.preprocessing.image import img_to_array

from ulitities.base_functions import get_file

# seed = 1
# np.random.seed(seed)

img_w = 256
img_h = 256

valid_labels=[0,1,2]

# FLAG_BINARY = False
FLAG_BINARY = True


input_path = '../../data/originaldata/sat_urban_4bands/'


output_path = '../../data/traindata/sat_urban_4bands/'



def creat_dataset_binary(in_path, out_path, image_num=50000, mode='original'):
    print('\ncreating dataset...')

    label_files,tt = get_file(os.path.join(in_path,'label/'))
    assert(tt!=0)

    image_each = image_num/len(label_files)

    print("\n1: produce road labels---------------------")
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

        X_height = dataset.RasterYSize
        X_width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        src_img = dataset.ReadAsArray(0, 0, X_width, X_height)
        # src_img = img_to_array(src_img)

        del dataset

        index = np.where(label_img == 1)  # 1: roads
        road_label = np.zeros((X_height, X_width), np.uint8)
        road_label[index] = 1

        print(np.unique(road_label))
        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = road_label[random_height: random_height + img_h, random_width: random_width + img_w]

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

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/roads/visualize/%d.png' % g_count), visualize)
            # cv2.imwrite((out_path + '/roads/src/%d.png' % g_count), src_roi)
            cv2.imwrite((out_path + '/roads/label/%d.png' % g_count), label_roi)

            src_sample_file = out_path + '/roads/src/%d.png' % g_count
            driver = gdal.GetDriverByName("GTiff")
            # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
            outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_Byte)
            if im_bands ==1:
                outdataset.GetRasterBand(1).WriteArray(src_roi)
            else:
                for i in range(im_bands):
                    outdataset.GetRasterBand(i+1).WriteArray(src_roi[i])
            del outdataset


            count += 1
            g_count += 1


    print("\n2: produce buildings labels---------------------")

    g_count=0

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

        X_height = dataset.RasterYSize
        X_width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        src_img = dataset.ReadAsArray(0, 0, X_width, X_height)
        # src_img = img_to_array(src_img)

        del dataset

        index = np.where(label_img == 2)  # 1: buildings
        building_label = np.zeros((X_height, X_width), np.uint8)
        building_label[index] = 1

        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = building_label[random_height: random_height + img_h, random_width: random_width + img_w]

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

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/buildings/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/buildings/label/%d.png' % g_count), label_roi)

            src_sample_file = out_path + '/buildings/src/%d.png' % g_count
            driver = gdal.GetDriverByName("GTiff")
            # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
            outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_Byte)
            if im_bands == 1:
                outdataset.GetRasterBand(1).WriteArray(src_roi)
            else:
                for i in range(im_bands):
                    outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
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


    if FLAG_BINARY == True:
        output_path = ''.join([output_path, '/binary/'])
    else:
        output_path = ''.join([output_path, '/multiclass/'])


    if FLAG_BINARY==True:
        print("Produce labels for binary classification")
        creat_dataset_binary(input_path, output_path, 20000, mode='augment')
    # else:
    #     print("produce labels for multiclass")
    #     creat_dataset_multiclass(input_path, output_path, 200000, mode='augment')







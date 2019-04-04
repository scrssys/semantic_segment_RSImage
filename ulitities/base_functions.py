
#coding:utf-8
import os
import sys
import cv2
import numpy as np
import gdal
import gc


UINT8=0
UINT10 =1
UINT16=2

class Base_ulitities:
    def __init__(self):
        return 0

    def load_img_by_cv2(self, path, grayscale=False):

        """

        :param path: input image file path
        :param grayscale:  bool value
        :return: flag, image values
        """
        if not os.path.isfile(path):
            print("input path is not a file!")
            return -1, None
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
        return 0, img





def load_img_by_cv2(path, grayscale=False):
    """

    :param path: input image file path
    :param grayscale:  bool value
    :return: flag, image values
    """
    if not os.path.isfile(path):
        print("input is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    return 0, img

def load_img_normalization_by_cv2(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")/255.0
    return 0, img



def load_img_by_gdal(path, grayscale=False):
    dataset = gdal.Open(path)
    assert(dataset is not None)

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0,0,x_width,y_height)
    geotransform = dataset.GetGeoTransform()

    if grayscale == False:
        img = np.array(img, dtype="float")
        if im_bands>1:
            img = np.transpose(img, (1,2,0))
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset

    return img
def load_img_by_gdal_geo(path, grayscale=False):
    dataset = gdal.Open(path)
    assert(dataset is not None)

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0,0,x_width,y_height)
    geotransform = dataset.GetGeoTransform()

    if grayscale == False:
        img = np.array(img, dtype="float")
        if im_bands>1:
            img = np.transpose(img, (1,2,0))
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset

    return img, geotransform

def load_img_by_gdal_blocks(path, x,y,width,height,grayscale=False):
    dataset = gdal.Open(path)
    assert(dataset is not None)

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    if y+height>y_height:
        height = y_height-y
    img = dataset.ReadAsArray(x,y,width,height)
    if grayscale == False:
        img = np.array(img, dtype="float")
        if im_bands>1:
            img = np.transpose(img, (1,2,0))
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset
    gc.collect()

    return img



def load_img_normalization(input_bands, path, data_type=UINT8):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if input_bands==1 and data_type==UINT8:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif input_bands ==3 and data_type==UINT8:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    else:
        img = load_img_by_gdal(path)
        if data_type==UINT8:
            img = img/255.0
        elif data_type==UINT10:
            img = img / 1024.0
        elif data_type == UINT16:
            img = img / 65535.0
        else:
            print("Not recognize this type!")
            sys.exit(-1)
        img = np.clip(img, 0.0, 1.0)
    return 0, img


def get_file(file_dir, file_type='.png'):
    """

    :param file_dir: directory of input files, it may have sub_folders
    :param file_type: file format, namely postfix
    :return: L: a list of files under the file_dir and sub_folders; num: the length of L
    """
    L=[]
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1]==file_type:
                L.append(os.path.join(root,file))
    num = len(L)
    return L, num

""" check the size of src_img and label_img"""
def compare_two_image_size(img_one, img_two, grayscale=False):
    if grayscale:
        h1, w1 = img_one.shape
        h2, w2 = img_two.shape
        assert(h1==h2 and w1==w2)
    else:
        h1, w1, _ = img_one.shape
        h2, w2, _ = img_two.shape
        assert (h1 == h2 and w1 == w2)


#coding:utf-8

import os
import sys
import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from ulitities.base_functions import get_file

NoData = 65535
# input_path = '../../data/originaldata/zs/test/original/'
input_path = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/ducha/DCtuitiantu/original_src/'

# output_path = '/home/omnisky/PycharmProjects/data/originaldata/4bands/test/'
# output_path = '../../data/originaldata/zs/test/stretched/'
output_path = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/ducha/DCtuitiantu/src/'
absname = 'lizhou_test_4bands.png'  # fenyi11, qingbaijiang, yujiang4, lizhou_test_4bands


def convert_single_image():
    filename = os.path.join(input_path, absname)
    if not os.path.isfile(filename):
        print("input file dose not exist:{}\n".format(filename))
        sys.exit(-1)

    print(filename)
    dataset = gdal.Open(filename)

    height = dataset.RasterYSize
    width = dataset.RasterXSize
    im_bands = dataset.RasterCount

    img = dataset.ReadAsArray(0,0,width, height)
    del dataset
    img = np.array(img, np.uint16)
    result =[]
    for i in range(im_bands):
        data = np.array(img[i])
        maxium = data.max()
        minm = data.min()
        mean = data.mean()
        std = data.std()

        data = data.reshape(height*width)
        ind = np.where((data > 0) & (data <NoData))
        ind = np.array(ind)
        # ind = ind.sort()
        ind = np.sort(ind)
        a, b = ind.shape
        print(b)
        tmp = np.zeros(b, np.uint16)
        for j in range(b):
            tmp[j] = data[ind[0,j]]

        tmaxium = tmp.max()
        tminm = tmp.min()
        tmean = tmp.mean()
        tstd = tmp.std()

        tt = (data-tmean)/tstd # first Z-score normalization
        tt = (tt+4)*200/8.0    # second min-max normalization to 255
        tind = np.where(data==0)

        tt = np.array(tt)
        tt = tt.astype(np.uint8)
        # tt = tt.astype(np.uint16)
        tt[tind] = 0

        out = tt.reshape((height, width))
        result.append(out)

        # plt.imshow(out)
        # plt.show()
        # cv2.imwrite((output_path + '%d.png' % i),out)


    outputfile = os.path.join(output_path, absname)
    driver = gdal.GetDriverByName("GTiff")

    outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_Byte)
    # outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)
    # if im_bands ==1:
    #     outdataset.GetRasterBand(1).WriteArray(result[0])
    # else:
    for i in range(im_bands):
        # outdataset.GetRasterBand(i+1).WriteArray(result[i])
        outdataset.GetRasterBand(i + 1).WriteArray(result[i])

    del outdataset


def convert_all_image_to_8bits():
    src_files, tt = get_file(input_path)
    assert (tt != 0)

    for file in tqdm(src_files):

        absname = os.path.split(file)[1]
        print(absname)
        if not os.path.isfile(file):
            print("input file dose not exist:{}\n".format(file))
            continue
            # sys.exit(-1)

        dataset = gdal.Open(file)
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        img = dataset.ReadAsArray(0, 0, width, height)
        del dataset
        img = np.array(img, np.uint16)
        result = []
        for i in range(im_bands):
            data = np.array(img[i])
            maxium = data.max()
            minm = data.min()
            mean = data.mean()
            std = data.std()
            data = data.reshape(height * width)
            ind = np.where((data > 0) & (data <NoData))
            ind = np.array(ind)
            # ind = ind.sort()
            #         ind = np.sort(ind)
            a, b = ind.shape
            print(b)
            tmp = np.zeros(b, np.uint16)
            for j in range(b):
                tmp[j] = data[ind[0, j]]
            tmaxium = tmp.max()
            tminm = tmp.min()
            tmean = tmp.mean()
            tstd = tmp.std()
            tt = (data - tmean) / tstd  # first Z-score normalization
            # tt = (tt + 4) * 200 / 8.0  # second min-max normalization to 255
            tt = (tt + 4) * 255 / 8.0 - 255 # second min-max normalization to 255

            tind = np.where(data == 0)

            tt = np.array(tt)
            tt = tt.astype(np.uint8)
            # tt = tt.astype(np.uint16)
            tt[tind] = 0

            out = tt.reshape((height, width))
            result.append(out)

        # plt.imshow(out)
        # plt.show()
        # cv2.imwrite((output_path + '%d.png' % i),out)

        # absname = os.path.split(file)[1]

        outputfile = os.path.join(output_path, absname)
        driver = gdal.GetDriverByName("GTiff")

        outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_Byte)
        # outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)
        # if im_bands ==1:
        #     outdataset.GetRasterBand(1).WriteArray(result[0])
        # else:
        for i in range(im_bands):
            outdataset.GetRasterBand(i + 1).WriteArray(result[i])

        del outdataset


def convert_all_image_to_16bits():
    # src_files, tt = get_file(input_path,file_type='.tif')
    src_files, tt = get_file(input_path)
    assert (tt != 0)

    for file in tqdm(src_files):

        absname = os.path.split(file)[1]
        absname = absname.split('.')[0]
        # absname = 'shuidao.png'
        absname = ''.join([absname, '.png'])
        print(absname)
        if not os.path.isfile(file):
            print("input file dose not exist:{}\n".format(file))
            # sys.exit(-1)
            continue

        dataset = gdal.Open(file)
        if dataset == None:
            print("Open file failed: {}".format(file))
            continue

        height = dataset.RasterYSize
        width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        im_type = dataset.GetRasterBand(1).DataType
        img = dataset.ReadAsArray(0, 0, width, height)
        del dataset
        # img = np.array(img, np.uint16)
        img = np.array(img, np.float32)
        result = []
        for i in range(im_bands):
            if im_bands==1:
                data = img
            else:
                data = np.array(img[i])
            maxium = data.max()
            minm = data.min()
            mean = data.mean()
            std = data.std()
            print(maxium, minm, mean, std)
            data = data.reshape(height * width)
            ind = np.where((data > 0) & (data < NoData))
            ind = np.array(ind)

            a, b = ind.shape
            print("valid value number: {}\n".format(b))
            # tmp = np.zeros(b, np.uint16)
            tmp = np.zeros(b, np.float32)
            for j in range(b):
                tmp[j] = data[ind[0, j]]
            tmaxium = tmp.max()
            tminm = tmp.min()
            tmean = tmp.mean()
            tstd = tmp.std()
            print(tmaxium, tminm, tmean, tstd)
            tt = (data - tmean) / tstd  # first Z-score normalization
            tt = (tt + 4) * 1024 / 8.0-100
            tind = np.where(data == 0)

            tt = np.array(tt)
            # tt = tt.astype(np.uint8)
            tt = tt.astype(np.uint16)
            tt[tind] = 0

            smaxium = tt.max()
            sminm = tt.min()
            smean = tt.mean()
            sstd = tt.std()
            print(smaxium, sminm, smean, sstd)

            out = tt.reshape((height, width))
            result.append(out)


        outputfile = os.path.join(output_path, absname)
        driver = gdal.GetDriverByName("GTiff")

        outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)

        for i in range(im_bands):
            outdataset.GetRasterBand(i + 1).WriteArray(result[i])

        del outdataset

if __name__ == '__main__':
    # convert_all_image_to_8bits()
    convert_all_image_to_16bits()
    # convert_single_image()


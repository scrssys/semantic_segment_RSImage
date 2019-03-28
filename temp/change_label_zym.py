import os, sys, gdal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from ulitities.base_functions import get_file, load_img_by_gdal


def write_img_by_gdal(path, data, bands, dtype):
    data = np.array(data)
    if bands >1:
        a,b,c = data.shape
        if c<a:
            data= data.transpose(2,0,1)
        bands, img_w, img_h = data.shape
    else:
        img_w, img_h = data.shape
    driver = gdal.GetDriverByName("GTiff")
    outdataset = driver.Create(path, img_w,img_h, bands, dtype)

    if outdataset == None:
        print("create dataset failed!\n")
        sys.exit(-2)
    if bands == 1:
        outdataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(bands):
            outdataset.GetRasterBand(i + 1).WriteArray(data[i])
    del outdataset


input_dir = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/rice/original_label/'
output_dir = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/rice/label/'


if __name__=='__main__':

    if not os.path.isdir(input_dir):
        print("input dir is not existed: {}".format(input_dir))
        sys.exit(-1)

    files,numb = get_file(input_dir)

    for file in tqdm(files):
        absname = os.path.split(file)[1]
        absname = absname.split('.')[0]
        tmp_file = ''.join([output_dir, absname, '.png'])
        # img = load_img_by_gdal(file, grayscale=True)
        img = cv2.imread(file,0)
        # img = np.array(img, np.uint8)
        print("original value: {}".format(np.unique(img)))

        size = img.shape

        result = np.zeros(size, np.uint8)
        ind_targt = np.where(img==1)
        result[ind_targt]= 1

        ind_nodata = np.where(img==255)
        result[ind_nodata]=127
        print("new value: {}".format(np.unique(result)))

        # plt.imshow(result)
        # plt.show()

        # write_img_by_gdal(tmp_file,result, 1, 1)
        cv2.imwrite(tmp_file, result)
        print("Done:{}\n".format(tmp_file))










import cv2
import gdal
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


img_path = '../../data/originaldata/sat4bands/src/sample1.png'
outdir_rgb = '../../data/originaldata/SatRGB/src/'
outdir_nrg = '../../data/originaldata/SatNRG/src/'

if __name__=='__main__':

    dataset = gdal.Open(img_path)
    if dataset ==None:
        print("file open failed\n")
        sys.exit(-1)

    img_bands=dataset.RasterCount
    assert (img_bands >= 4)

    img = dataset.ReadAsArray(0, 0)
    rgb_img = img[:3]
    rgb_img = np.transpose(rgb_img, (1, 2, 0))

    p_band=dataset.GetRasterBand(1)
    data_type = p_band.DataType
    # data_type = gdal.GetDataTypeByName(p_band.DataType)
    if data_type==1:

    elif data_type==2:
        # img = dataset.ReadAsArray(0, 0)
        # rgb_img = img[:3]
        # rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = rgb_img*255/6
        rgb_img = rgb_img.astype(np.uint8)
        plt.imshow(rgb_img, cmap='jet')
        plt.show()
        rgb_filename = ''.join([outdir_rgb, os.path.split(img_path)[1]])

        cv2.imwrite(rgb_filename, rgb_img)

        nrg_img = img[1:4]
        nrg_img = np.transpose(nrg_img, (1, 2, 0))
        nrg_img = nrg_img.astype(np.uint8)
        plt.imshow(nrg_img)
        plt.show()
        rgb_filename = ''.join([outdir_nrg, os.path.split(img_path)[1]])

        cv2.imwrite(rgb_filename, nrg_img)





    print("complete!\n")


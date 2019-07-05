
import gdal
import os, sys
import cv2
import operator
from libtiff import TIFF
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ulitities.base_functions import get_file, load_img_by_gdal
# class_types=[[0,0,0],[0,200,0], [150,250,0],[150,200,150],[200,0,200],[150,0,250], [150,150,250],[250,200,0],[200,200,0],[200,0,0],[250,0,150],
#        [200,150,150],[250,150,150],[0,0,200],[0,150,200],[0,200,250]]

input_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/test/original_images'
output_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/test/forclass_images'

if __name__=="__main__":
    print("INFO: Starting ........")
    files,_=get_file(input_dir)

    for file in tqdm(files):
        file_name = os.path.split(file)[1]
        # img=load_img_by_gdal(file)
        tif=TIFF.open(file, mode='r')
        img=tif.read_image()
        img=np.array(img)
        tif.close()

        plt.imshow(img[:,:,3])
        plt.show()

        output_file = os.path.join(output_dir, file_name)
        driver = gdal.GetDriverByName("GTiff")
        # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
        outdataset = driver.Create(output_file, img.shape[1], img.shape[0], img.shape[2], gdal.GDT_Byte)
        if img.shape[2] == 1:
            outdataset.GetRasterBand(1).WriteArray(img)
        else:
            for i in range(img.shape[2]):
                outdataset.GetRasterBand(i + 1).WriteArray(img[:,:,i])
        del outdataset


    print("done")




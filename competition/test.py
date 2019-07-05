import gdal
import os, sys
import cv2
import operator
from libtiff import TIFF
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ulitities.base_functions import get_file,load_img_by_gdal

input_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/samples/transformed/src'

if __name__=="__main__":
    print("INFO: Starting ........")
    files,_=get_file(input_dir)

    for file in tqdm(files):
        file_name = os.path.split(file)[1]
        img=load_img_by_gdal(file)
        # tif=TIFF.open(file, mode='r')
        # img=tif.read_image()
        # img=np.array(img)
        # tif.close()

        plt.imshow(img[:,:,3])
        plt.show()

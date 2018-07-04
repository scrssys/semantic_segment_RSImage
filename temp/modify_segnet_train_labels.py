#coding:utf-8

import sys
import os
import numpy as np
import cv2
from tqdm import tqdm

from ulitities.base_functions import load_img, get_file

input_path = '../../data//traindata/segnet/label_old/'
output_path = '../../data//traindata/segnet/label/'
label_class=[0,1,2] # remain only road, buildings

height=256
width=256


if __name__=='__main__':

    files = get_file(input_path)
    for filename in tqdm(files):
        ret, img = load_img(filename, grayscale=True)
        assert(ret == 0)
        for i in range(height):
            for j in range(width):
                if img[i,j]==4:
                    img[i,j] =1
                elif img[i,j]==2:
                    img[i,j]=2
                else:
                    print ("\n img[{},{}]:{}".format(i,j,img[i,j]))
                    img[i,j]=0

        cv2.imwrite(output_path+os.path.split(filename)[1],img)




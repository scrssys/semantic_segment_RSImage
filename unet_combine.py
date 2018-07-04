#coding:utf-8

import os
import cv2
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from ulitities.base_functions import load_img

FOREGROUND = 1 # define the foreground value

input_path = '../data/predict/unet/'
output_file = '../data/predict/result_unet_1.png'

mask_pool = ['mask_road_img_1.png','mask_buildings_img_1.png']


def check_input_file():
    ret, img_1 = load_img(input_path+mask_pool[0],grayscale=True)
    assert (ret == 0)

    height, width = img_1.shape
    num_img = len(mask_pool)

    for next_index in range(1,num_img):
        next_ret, next_img=load_img(input_path+mask_pool[next_index],grayscale=True)
        assert (next_ret ==0 )
        next_height, next_width = next_img.shape
        assert(height==next_height and width==next_width)
    return height, width



def combine_all_mask(height, width,input_path,mask_pool):
    final_mask=np.zeros((height,width),np.uint8)
    for idx,file in enumerate(mask_pool):
        ret,img = load_img(input_path+file,grayscale=True)
        assert (ret == 0)
        label_value=0
        if 'road' in file:
            label_value =1
        elif 'buildings' in file:
            label_value=2
        # label_value = idx+1
        for i in tqdm(range(height)):
            for j in range(width):
                if img[i,j]==FOREGROUND:
                    if label_value==2:
                        final_mask[i,j]=label_value
                    else:
                        final_mask[i,j]=label_value
    return final_mask



if __name__=='__main__':

    x,y=check_input_file()

    result_mask=combine_all_mask(x,y,input_path,mask_pool)

    plt.imshow(result_mask, cmap='gray')
    plt.title("final mask")
    plt.show()

    cv2.imwrite(output_file,result_mask)
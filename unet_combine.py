#coding:utf-8

import os
import cv2
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from ulitities.base_functions import load_img

FOREGROUND = 40# for segnet:40; for unet=125; define the foreground value

ROAD_VALUE=125
BUILDING_VALUE=255

"""for unet"""
# input_path = '../data/predict/unet/'
# output_file = '../data/predict/result_unet_combined.png'
#
# mask_pool = ['mask_unet_roads_1.png','mask_unet_buildings_1.png']

"""for segnet"""
input_path = '../data/predict/segnet/'
output_file = '../data/predict/result_segnet_combined.png'
mask_pool = ['mask_segnet_building.png','mask_segnet_road.png']


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
    """

    :param height:
    :param width:
    :param input_path:
    :param mask_pool:
    :return:

    prior: bulidings(2)>road(1)
    """
    final_mask=np.zeros((height,width),np.uint8)
    for idx,file in enumerate(mask_pool):
        ret,img = load_img(input_path+file,grayscale=True)
        assert (ret == 0)
        label_value=0
        if 'road' in file:
            label_value =ROAD_VALUE
        elif 'building' in file:
            label_value=BUILDING_VALUE
        # label_value = idx+1
        for i in tqdm(range(height)):
            for j in range(width):
                if img[i,j]>=FOREGROUND:
                    print ("img[{},{}]:{}".format(i,j,img[i,j]))
                    if label_value==ROAD_VALUE:
                        final_mask[i,j]=label_value
                    elif label_value==BUILDING_VALUE:
                        print ("final_mask[{},{}]:{}".format(i, j, final_mask[i, j]))
                        if final_mask[i,j]!=ROAD_VALUE:
                            final_mask[i,j]=label_value

    return final_mask



if __name__=='__main__':

    x,y=check_input_file()

    result_mask=combine_all_mask(x,y,input_path,mask_pool)

    plt.imshow(result_mask, cmap='gray')
    plt.title("final mask")
    plt.show()

    cv2.imwrite(output_file,result_mask)
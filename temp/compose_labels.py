#coding:utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ulitities.base_functions import load_img, compare_two_image_size

road_label_file = '../../data/originaldata/unet/roads/label/5.png'
building_label_file = '../../data/originaldata/unet/buildings/label/5.png'
output_file = '../../data/originaldata/all/label/5.png'


if __name__=='__main__':

    ret1, road_label = load_img(road_label_file, grayscale=True)
    assert(ret1==0)

    ret2, building_label = load_img(building_label_file, grayscale=True)
    assert (ret2== 0)

    compare_two_image_size(road_label, building_label, grayscale=True)

    height, width = road_label.shape

    output_label = np.zeros((height,width), np.uint8)
    index = np.where(road_label==1)
    output_label[index]=1
    idx = np.where(building_label == 1)
    output_label[idx] = 2

    plt.imshow(output_label, cmap='gray')
    plt.show()

    cv2.imwrite(output_file,output_label)




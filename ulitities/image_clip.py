#coding:utf-8

from PIL import Image
import cv2
import numpy as np
from base_functions import load_img
import matplotlib.pyplot as plt


input_src_file = '/home/omnisky/PycharmProjects/data/originaldata/sat4bands/src/ruoergai_2_321band.png'

clip_src_file = '../../data/test/GF2_changning_1.png'

window_size = 2048

if __name__=='__main__':
    ret, img =load_img(input_src_file)
    assert (ret==0)

    height,width,_ = img.shape


    x = np.random.randint(0, height-window_size-1)
    y = np.random.randint(0, width - window_size - 1)

    output_img = img[x:x+window_size, y:y+window_size,:]

    plt.imshow(output_img)
    plt.show()


    cv2.imwrite(clip_src_file, output_img)

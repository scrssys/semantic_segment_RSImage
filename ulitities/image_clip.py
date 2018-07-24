#coding:utf-8

from PIL import Image
import cv2
import numpy as np
from base_functions import load_img


input_file = '../../data/originaldata/all/src/H48G036036.png'

output_test_file = '../../data/test/H48G036036_1.png'

window_size = 2048

if __name__=='__main__':
    # img = cv2.imread(input_file)
    ret, img =load_img(input_file)
    assert (ret==0)

    height,width,_ = img.shape


    x = np.random.randint(0, height-window_size-1)
    y = np.random.randint(0, width - window_size - 1)

    output_img = img[x:x+window_size, y:y+window_size,:]

    cv2.imwrite(output_test_file, output_img)

#coding:utf-8

import numpy as np
import sys
import os
import cv2
from tqdm import tqdm
from ulitities.base_functions import load_img, get_file

FLAG_USING_UNET = True
segnet_labels = [0, 1, 2] # have not test for segnetlabels
unet_labels = [0, 1]

input_src_path = '../../data/originaldata/unet/roads/src/'
input_label_path = '../../data/originaldata/unet/roads/label/'

HAS_INVALID_VALUE = False


def make_label_valid(img, true_values):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            tmp = img[i,j]
            if not tmp in true_values:
                print("img[{},{}]: {}".format(i,j,tmp))
                img[i,j]=0
    return img


if __name__ == '__main__':
    src_files,num = get_file(input_src_path)
    assert (num!=0)

    valid_labels = []
    if FLAG_USING_UNET:
        valid_labels = unet_labels
    else:
        valid_labels = segnet_labels

    for src_file in tqdm(src_files):
        label_file = input_label_path + os.path.split(src_file)[1]

        ret,src_img = load_img(src_file)
        assert(ret==0)

        ret,label_img = load_img(label_file, grayscale=True)
        assert (ret == 0)

        local_labels = np.unique(label_img)

        for tmp in local_labels:
            if tmp not in valid_labels:
                print ("\nWarning: some label is not valid value")
                print ("\nFile: {}".format(label_file))
                HAS_INVALID_VALUE = True


        if HAS_INVALID_VALUE == True:
            new_label_img = make_label_valid(label_img, valid_labels)
            cv2.imwrite(label_file, new_label_img)
            HAS_INVALID_VALUE = False






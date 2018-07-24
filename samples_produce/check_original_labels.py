#coding:utf-8

import numpy as np
import sys
import os
import cv2
from tqdm import tqdm
from ulitities.base_functions import load_img, get_file

import matplotlib.pyplot as plt


# segnet_labels = [0, 1, 2] # have not test for segnetlabels
# unet_labels = [0, 1]


# """for unet"""
# input_src_path = '../../data/originaldata/unet/roads/src/'
# input_label_path = '../../data/originaldata/unet/roads/label/'

# """for segnet"""
# input_src_path = '../../data/originaldata/segnet/src/'
# input_label_path = '../../data/originaldata/segnet/label/'


input_label_path = '../../data/originaldata/all/label/'
valid_labels = [0, 1, 2]

HAS_INVALID_VALUE = False
# FLAG_BINARY_LABELS = True


def make_label_valid(img, false_values):
    height, width = img.shape

    tp_img = img.reshape((height*width))
    for inv_lab in false_values:
        index = np.where(tp_img==inv_lab)
        tp_img[index]=0
    tp_img = tp_img.reshape((height, width))

    return tp_img



    #
    # for i in range(height):
    #     for j in range(width):
    #         tmp = img[i,j]
    #         if not tmp in true_values:
    #             print("img[{},{}]: {}".format(i,j,tmp))
    #             img[i,j]=0
    # return img


if __name__ == '__main__':
    files,num = get_file(input_label_path)
    assert (num!=0)

    # valid_labels = []
    # if FLAG_USING_UNET:
    #     valid_labels = unet_labels
    # else:
    #     valid_labels = segnet_labels

    for label_file in tqdm(files):
        # label_file = input_label_path + os.path.split(src_file)[1]
        #
        # ret,src_img = load_img(src_file)
        # assert(ret==0)

        ret,label_img = load_img(label_file, grayscale=True)
        assert (ret == 0)

        local_labels = np.unique(label_img)
        invalid_labels=[]

        for tmp in local_labels:
            if tmp not in valid_labels:
                invalid_labels.append(tmp)
                print ("\nWarning: some label is not valid value")
                print ("\nFile: {}".format(label_file))
                HAS_INVALID_VALUE = True


        if HAS_INVALID_VALUE == True:
            new_label_img = make_label_valid(label_img, invalid_labels)
            new_label_file = os.path.split(label_file)[0]+'/new_'+os.path.split(label_file)[1]
            cv2.imwrite(new_label_file, new_label_img)
            HAS_INVALID_VALUE = False
            label_img = new_label_img

        plt.imshow(label_img, cmap='gray')
        plt.show()






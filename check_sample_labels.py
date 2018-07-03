#coding:utf-8

import numpy as np
import sys
import os
import cv2
from tqdm import tqdm

FLAG_USING_UNET = True
segnet_labels = [0, 1, 2, 3, 4]
unet_labels = [0, 1]

input_src_path = '../data/originaldata/unet/roads/src/'
input_label_path = '../data/originaldata/unet/roads/label/'

HAS_INVALID_VALUE = False

def get_file(file_path, file_type='.png'):
    L=[]
    for root,dirs,files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1]==file_type:
                L.append(os.path.join(root,file))
    return L

def load_img(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        sys.exit(-1)
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")
    return img



def make_label_valid(img, valid_values):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            tmp = img[i,j]
            if not tmp in valid_labels:
                print("img[{},{}]: {}".format(i,j,tmp))
                img[i,j]=0
    return img


if __name__ == '__main__':
    src_files = get_file(input_src_path)

    valid_labels = [0, 1]
    if FLAG_USING_UNET:
        valid_labels = unet_labels
    else:
        valid_labels = segnet_labels

    for src_file in tqdm(src_files):
        label_file = input_label_path + os.path.split(src_file)[1]
        src_img = load_img(src_file)
        label_img = load_img(label_file, grayscale=True)
        local_labels = np.unique(label_img)

        for tmp in local_labels:
            if not tmp in valid_labels:
                print ("\nWarning: some label is not valid value")
                print ("\nFile: {}".format(label_file))
                HAS_INVALID_VALUE = True
                # label_img = make_label_valid(label_img,valid_labels)

        if HAS_INVALID_VALUE == True:
            label_img = make_label_valid(label_img, valid_labels)
            cv2.imwrite(label_file, label_img)
            HAS_INVALID_VALUE = False






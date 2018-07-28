
#coding:utf-8
import os
import cv2
import numpy as np


def load_img(path, grayscale=False):
    """

    :param path: input image file path
    :param grayscale:  bool value
    :return: flag, image values
    """
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    return 0, img

def load_img_normalization(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")/255.0
    return 0, img



def get_file(file_dir, file_type='.png'):
    """

    :param file_dir: directory of input files, it may have sub_folders
    :param file_type: file format, namely postfix
    :return: L: a list of files under the file_dir and sub_folders; num: the length of L
    """
    L=[]
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1]==file_type:
                L.append(os.path.join(root,file))
    num = len(L)
    return L, num

""" check the size of src_img and label_img"""
def compare_two_image_size(img_one, img_two, grayscale=False):
    if grayscale:
        h1, w1 = img_one.shape
        h2, w2 = img_two.shape
        assert(h1==h2 and w1==w2)
    else:
        h1, w1, _ = img_one.shape
        h2, w2, _ = img_two.shape
        assert (h1 == h2 and w1 == w2)


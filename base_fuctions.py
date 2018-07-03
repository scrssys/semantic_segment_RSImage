
#coding:utf-8
import os
import cv2
import numpy as np


def load_img(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1,None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        # img = np.array(img, dtype="float")
    return 0,img
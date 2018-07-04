
#coding:utf-8
import os
import cv2
import numpy as np


def load_img(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        # img = np.array(img, dtype="float")
    return 0, img

def get_file(file_path, file_type='.png'):
    L=[]
    for root,dirs,files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1]==file_type:
                L.append(os.path.join(root,file))
    return L
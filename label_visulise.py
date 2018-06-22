#coding:utf-8

import cv2
import sys, os
import numpy as np

input_path = './data/traindata/unet/water/label/'
output_path = './data/traindata/unet/water/visulise/'


def get_files(file_dir, format='.png'):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == format:
                L.append(os.path.join(root, file))
    return L

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img

if __name__ == '__main__':
    srcfiles = get_files(input_path)
    for index, file in enumerate(srcfiles):
        img = load_img(file,grayscale=True)
        img = img*100
        filename = os.path.split(file)[1]
        # print (filename)
        outfile = output_path + filename
        print (outfile)

        cv2.imwrite(outfile, img)

        # sys.exit()

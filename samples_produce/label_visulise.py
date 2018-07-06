#coding:utf-8

import cv2
import sys, os
from ulitities.base_functions import load_img,get_file

input_path = '../../data/traindata/unet/buildings/label/'
output_path = '../../data/traindata/unet/buildings/visulize/'


if __name__ == '__main__':

    if not os.path.isdir(input_path):
        print("No input directory:{}".format(input_path))
        sys.exit(-1)
    if not os.path.isdir(output_path):
        print("No output directory:{}".format(output_path))
        os.mkdir(output_path)

    srcfiles, tt= get_file(input_path)
    assert(tt!=0)

    for index, file in enumerate(srcfiles):
        ret,img = load_img(file,grayscale=True)
        assert(ret==0)

        img = img*100
        filename = os.path.split(file)[1]
        outfile = os.path.join(output_path,filename)
        print (outfile)

        cv2.imwrite(outfile, img)


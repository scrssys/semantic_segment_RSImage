

import os, sys, gdal
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from ulitities.base_functions import get_file, load_img_by_cv2

inputdir = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/test/paper/label/'
outputdir  = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/test/paper/pred_V2019/label_building/'


if __name__=="__main__":

    files, _= get_file(inputdir)

    for file in tqdm(files):
        _,img = load_img_by_cv2(file, grayscale=True)
        absname = os.path.split(file)[1]
        a,b = img.shape
        data = np.zeros((a,b),np.uint8)
        index = np.where(img==2)
        data[index]=1
        plt.imshow(data)
        plt.show()

        outfile = os.path.join(outputdir,absname)
        cv2.imwrite(outfile, data)
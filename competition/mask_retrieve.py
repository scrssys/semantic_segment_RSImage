
import os, sys
import gdal
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ulitities.base_functions import get_file

class_types = [[0, 0, 0], [0, 200, 0], [150, 250, 0], [150, 200, 150], [200, 0, 200], [150, 0, 250], [150, 150, 250],
               [250, 200, 0], [200, 200, 0], [200, 0, 0], [250, 0, 150],
               [200, 150, 150], [250, 150, 150], [0, 0, 200], [0, 150, 200], [0, 200, 250]]

input_dir='/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/test/pred/2019-08-16_09-30-55'
output_dir='/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/test/segmentation'

if __name__=='__main__':
    print("Info: starting to get final mask...")

    files,_=get_file(input_dir)
    for file in tqdm(files):
        file_name = os.path.split(file)[1]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        a,b = img.shape
        mask = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for i in range(len(class_types)):
            index = np.where(img==i)
            mask[index[0],index[1],0]=class_types[i][2]
            mask[index[0], index[1], 1] = class_types[i][1]
            mask[index[0], index[1], 2] = class_types[i][0]

        # plt.imshow(mask)
        # plt.show()

        output_file = os.path.join(output_dir, file_name)
        cv2.imwrite(output_file, mask)







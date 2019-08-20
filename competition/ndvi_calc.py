
import os, sys
from ulitities.base_functions import get_file, load_img_by_gdal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



class Not3D(Exception):
    pass


input_dir = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/samples/new/train/ori_src/'
output_dir='/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/rssrai2019/samples/new/train/with_ndvi'

if __name__=='__main__':

    files,_=get_file(input_dir)
    if len(files)==0:
        print("no file in input directory: {}".format(input_dir))
        sys.exit(-1)

    for file in tqdm(files):


        img = load_img_by_gdal(file)
        if len(img)==0:
            print("open file failed:{}".format(file))
            continue

        try:
            a,b,c=img.shape
        except:
            print("not 3 dimensional img")
            continue
        else:
            if c>a:
                img = np.transpose(img,(1,2,0))
                a,b,c=img.shape

        if c<4:
            print("input image should least 4 bands")
            continue

        result = np.((a,b,c+1),np.float16)




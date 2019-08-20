
import os,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ulitities.base_functions import load_img_by_gdal


inputfile = '/media/omnisky/b1aca4b8-81b8-4751-8dee-24f70574dae9/Global/traindata/4band_nodsm/t.png'


if __name__=="__main__":
    i_img = load_img_by_gdal(inputfile)

    i_img =np.asarray(i_img, np.uint16)
    img = i_img[:,:,0]

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    data = Image.fromarray(img, mode='I')
    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.show()


    res = np.asarray(data,np.uint16)
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()

    b_img = data.resize((240, 240), Image.BILINEAR)
    plt.figure()
    plt.imshow(b_img, cmap='gray')
    plt.show()

    res_1 = np.asarray(b_img, np.uint16)
    plt.figure()
    plt.imshow(res_1, cmap='gray')
    plt.show()

    os.system('pause')



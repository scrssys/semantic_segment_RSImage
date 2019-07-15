

import os,sys
import gdal, cv2
from ulitities.base_functions import get_file
from tqdm import tqdm
import numpy as np
input_dir = '/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/AerialImageDataset/train/gt/'
output_dir ="/media/omnisky/6b62a451-463c-41e2-b06c-57f95571fdec/Backups/data/AerialImageDataset/train/label/"

files ,_=get_file(input_dir)
for file in tqdm(files):
    try:
        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        dataset = gdal.Open(file)
        if dataset == None:
            print("could not open file")
            sys.exit(-2)
        im_band = dataset.RasterCount
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        data = dataset.ReadAsArray(0, 0, width, height)
        del dataset

        absname = os.path.split(file)[1]
        img = np.array(data)

        out_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        index=np.where(img==255)
        out_img[index]=1
        img = np.array(img)
        print("shape:{}".format(img.shape))
        print("value:{}".format(np.unique(img)))
        print("new label's value:{}".format(np.unique(out_img)))
        import matplotlib.pyplot as plt
        # plt.imshow(out_img)
        # plt.show()
        output_file = os.path.join(output_dir, absname)
        cv2.imwrite(output_file,out_img)
        print("saved to :{}".format(output_file))

    except Exception:
        raise Exception
        # print("Warning: it's a RGB image")

    else:
        print("gray image")



#



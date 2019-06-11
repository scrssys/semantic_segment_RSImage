
import cv2
import gdal,os, sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ulitities.base_functions import get_file, load_img_by_gdal


def image_normalize(input_dict):
    input_dir = input_dict["input_dir"]
    output_dir = input_dict["output_dir"]
    nodata = input_dict["NoData"]
    result_bits = input_dict["OutBits"]
    valid_range = input_dict["StretchRange"]
    cut_value = input_dict["CutValue"]

    src_files, tt = get_file(input_dir)
    assert (tt != 0)
    factor=4.0

    if '8' in result_bits:
        assert (valid_range < 256)
        factor = 6.0
    elif '16' in result_bits:
        assert (valid_range < 65536)
        factor = 4.0
    else:
        pass

    for file in tqdm(src_files):

        absname = os.path.split(file)[1]
        # absname = absname.split('.')[0]
        # absname = ''.join([absname, '.tif'])
        print(absname)
        if not os.path.isfile(file):
            print("input file dose not exist:{}\n".format(file))
            # sys.exit(-1)
            continue

        dataset = gdal.Open(file)
        if dataset == None:
            print("Open file failed: {}".format(file))
            continue

        height = dataset.RasterYSize
        width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        im_type = dataset.GetRasterBand(1).DataType
        img = dataset.ReadAsArray(0, 0, width, height)
        geotransform = dataset.GetGeoTransform()
        del dataset
        # img = np.array(img, np.uint16)
        img = np.array(img, np.float32)
        result = []
        for i in range(im_bands):
            data = np.array(img[i])
            maxium = data.max()
            minm = data.min()
            mean = data.mean()
            std = data.std()
            print("\nOriginal max, min, mean,std:[{},{},{},{}]".format(maxium, minm, mean, std))
            data = data.reshape(height * width)
            ind = np.where((data > 0) & (data < nodata))
            ind = np.array(ind)

            a, b = ind.shape
            print("valid value number: {}".format(b))
            # tmp = np.zeros(b, np.uint16)
            tmp = np.zeros(b, np.float32)
            for j in range(b):
                tmp[j] = data[ind[0, j]]
            tmaxium = tmp.max()
            tminm = tmp.min()
            tmean = tmp.mean()
            tstd = tmp.std()
            # print(tmaxium, tminm, tmean, tstd)
            tt = (data - tmean) / tstd  # first Z-score normalization
            tt = (tt + factor) * valid_range / (2*factor) - cut_value
            tind = np.where(data == 0)

            tt = np.array(tt)
            # tt = tt.astype(np.uint8)
            tt = tt.astype(np.uint16)
            tt[tind] = 0

            smaxium = tt.max()
            sminm = tt.min()
            smean = tt.mean()
            sstd = tt.std()
            # print(smaxium, sminm, smean, sstd)
            print("New max, min, mean,std:[{},{},{},{}]".format(smaxium, sminm, smean, sstd))

            out = tt.reshape((height, width))
            result.append(out)

        outputfile = os.path.join(output_dir, absname)
        driver = gdal.GetDriverByName("GTiff")


        if '8' in result_bits:
            outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_Byte)
            outdataset.SetGeoTransform(geotransform)
        elif '16' in result_bits:
            outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)
            outdataset.SetGeoTransform(geotransform)
        # outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)

        for i in range(im_bands):
            outdataset.GetRasterBand(i + 1).WriteArray(result[i])

        del outdataset

    return 0


def image_clip(input_dict):

    input_src_file = input_dict['input_file']
    if not os.path.isfile(input_src_file):
        print("input file is not existing!")
        sys.exit(-1)

    dataset = gdal.Open(input_src_file)
    if dataset == None:
        print("Open file failed:{}".format(input_src_file))
        sys.exit(-1)

    Yheight = dataset.RasterYSize
    Xwidth = dataset.RasterXSize
    im_bands = dataset.RasterCount
    d_type = dataset.GetRasterBand(1).DataType
    img = dataset.ReadAsArray(0, 0, Xwidth, Yheight)
    del dataset

    x = int(input_dict['x'])
    y = int(input_dict['y'])
    height = int(input_dict['row'])
    width = int(input_dict['column'])
    assert (width <= Xwidth and height <= Yheight)
    output_file = input_dict['output_file']

    if im_bands == 1:
        output_img = img[y:y + height, x:x + width]
        output_img = np.array(output_img, np.uint16)
        output_img = np.array(output_img, np.uint8)
        plt.imshow(output_img)
        plt.show()
        cv2.imwrite(output_file, output_img)  # for label clip
    else:
        output_img = img[:, y:y + height, x:x + width]
        plt.imshow(output_img[0])
        plt.show()
        driver = gdal.GetDriverByName("GTiff")
        # outdataset = driver.Create(clip_src_file, window_size, window_size, im_bands, d_type)
        outdataset = driver.Create(output_file, width, height, im_bands, d_type)
        if outdataset == None:
            print("create dataset failed!\n")
            sys.exit(-2)
        if im_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(output_img)
        else:
            for i in range(im_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(output_img[i])
        del outdataset

    return 0
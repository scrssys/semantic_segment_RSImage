#coding:utf-8

import os
import sys
import gdal
import numpy as np



root = '../../data/originaldata/zs/test/stretched/'
files = ['extract_bottom_16bit.png','dem_r05_b.png', 'Int_slope100_Resample05_b.png']


output_file =root+'composed.png'

if __name__=='__main__':
    all_data=[]
    file_path = root + files[0]
    if not os.path.isfile(file_path):
        print("File dose not exist:{}".format(file_path))
        sys.exit(-1)
    # print("deal file: {}".format(file_path))
    dataset = gdal.Open(file_path)
    if dataset == None:
        print("Open falied:{}".format(files[0]))
        sys.exit(-2)

    x = dataset.RasterXSize
    y = dataset.RasterYSize
    # im_band = dataset.RasterCount
    d_type = dataset.GetRasterBand(1).DataType
    del dataset
    all_bands=0

    for file in files:
        file_path = root+file
        if not os.path.isfile(file_path):
            print("File dose not exist:{}".format(file_path))
            sys.exit(-3)
        print("deal file: {}".format(file_path))
        dataset = gdal.Open(file_path)
        if dataset==None:
            print("Open falied:{}".format(file))
            continue
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        if x!=width or y!=height:
            print("Error: input files have different width and height\n")
            sys.exit(-4)
        im_band = dataset.RasterCount
        all_bands +=im_band
        im_type = dataset.GetRasterBand(1).DataType
        if d_type !=im_type:
            print("Error: data types are not the same!\n")
            sys.exit(-5)
        im_data = dataset.ReadAsArray(0,0,width,height)
        im_data = np.array(im_data)
        all_data.append(im_data)
        del dataset

    # all_data = np.array(all_data)
    # a,b,c =all_data.shape
    # print("allbands : {}".format(c))

    my_driver = gdal.GetDriverByName("GTiff")
    out_dataset = my_driver.Create(output_file, x, y, all_bands, d_type)
    which_band =1
    for i in range(len(all_data)):
        dims= len(all_data[i].shape)
        print("dimension :{}".format(dims))
        if dims <3:
            out_dataset.GetRasterBand(which_band).WriteArray(all_data[i])
            which_band +=1
        else:
            im_bands = all_data[i].shape[0]
            for j in range(im_bands):
                out_dataset.GetRasterBand(which_band).WriteArray(all_data[i][j])
                which_band +=1

    print("Saved to: {}".format(output_file))

    del out_dataset





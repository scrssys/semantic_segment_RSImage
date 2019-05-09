#coding:utf8
""""
    This is main procedure for remote sensing image semantic segmentation

"""
import cv2
import numpy as np
import os
import sys
import gc
import gdal
import argparse
from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('tf')
K.clear_session()
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

# from base_predict_functions import orignal_predict_notonehot, smooth_predict_for_binary_notonehot
from ulitities.base_functions import load_img_by_gdal,load_img_by_gdal_geo, load_img_by_gdal_blocks, UINT10,UINT8,UINT16, get_file, polygonize
from predict_backbone import predict_img_with_smooth_windowing,core_orignal_predict,core_smooth_predict_multiclass, core_smooth_predict_binary

from config_pred import Config_Pred
import pandas as pd

"""
   The following global variables should be put into meta data file 
"""
import  argparse
import json , time
parser=argparse.ArgumentParser(description='RS classification train')
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
parser.add_argument('--config', dest='config_file', help='json file to config',
                         default='config_pred_multiclass.json')
args=parser.parse_args()
gpu_id=args.gpu_id
print("gpu_id:{}".format(gpu_id))
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

config_file = args.config_file
print("cofig file:{}".format(config_file))
with open(args.config_file, 'r') as f:
    cfgl = json.load(f)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# with open("config.json", 'r') as f:
#     cfg = json.load(f)

config = Config_Pred(**cfgl)
print(config)

# sys.exit(-1)

im_type = UINT8
if "10" in config.im_type:
    im_type = UINT10
elif "16" in config.im_type:
    im_type=UINT16
else:
    pass

target_class =config.mask_classes
if target_class>1:   # multiclass, target class = total class -1
    if target_class==2:
        print("Warning: target classes should not be 2, this must be binary classification!")
        target_class =1
    else:
        target_class -=1

FLAG_APPROACH_PREDICT = 0 # 0: original predict, 1: smooth predict
if "smooth" in config.strategy:
    FLAG_APPROACH_PREDICT =1
else:
    pass

date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
output_dir = ''.join([config.mask_dir, '/',date_time])
os.mkdir(output_dir)

block_size = config.block_size
nodata = config.nodata


if __name__ == '__main__':
    input_files = []
    if os.path.isfile(config.img_input):
        print("[INFO] input is one file...")
        input_files.append(config.img_input)
    elif os.path.isdir(config.img_input):
        print("[INFO] input is a directory...")
        in_files, _ = get_file(config.img_input, config.suffix)
        for file in in_files:
            input_files.append(file)
    print("{} images will be classified".format(len(input_files)))

    # sys.exit(-1)
    csv_file = os.path.join(output_dir, 'readme.csv')
    df = pd.DataFrame(list(config))
    df.to_csv(csv_file)

    out_bands = config.mask_classes
    model = load_model(config.model_path)
    # print(model.summary())

    for img_file in tqdm(input_files):
        print("\n[INFO] opening image:{}...".format(img_file))
        abs_filename = os.path.split(img_file)[1]
        abs_filename = abs_filename.split(".")[0]
        whole_img, geoinf = load_img_by_gdal_geo(img_file)
        print("GeomTransform:{}".format(geoinf))
        H,W,C = np.array(whole_img).shape
        if C>1:
            nodata_indx = np.where(whole_img[:,:,0]==nodata)
        else:
            nodata_indx = np.where(whole_img == nodata)
        del whole_img
        gc.collect()

        nb_blocks = int(H*W/block_size)
        if H*W>nb_blocks*block_size:
            nb_blocks +=1
        block_h = int(block_size/W)
        print("single block size :[{},{}]".format(block_h,W))
        result_mask = np.zeros((H, W), np.uint8)
        for i in tqdm(list(range(nb_blocks))):
            start =block_h*i
            this_h = block_h
            if (i+1)*block_h>H:
                this_h = H-i*block_h
            end = start+this_h
            # b_img = load_img_by_gdal_blocks(img_file,0,start,W,this_h)
            b_img = load_img_by_gdal_blocks(img_file, 0, start, W, this_h+config.window_size)
            if i ==nb_blocks-1:
                exp_img = np.zeros((this_h+config.window_size, W, C), np.uint16)
                exp_img[:this_h,:,:] = b_img
            else:
                exp_img = b_img
                # exp_img = np.zeros((this_h+config.window_size, W, C), np.uint16)
                # exp_img[:, :, :] = b_img[:,:,:]
            # b_img = whole_img[start:end,:,:]
            # plt.imshow(b_img[:,:,1])
            # plt.show()
            # sys.exit(-3)
            if im_type == UINT8:
                input_img = exp_img / 255.0
            elif im_type == UINT10:
                input_img = exp_img / 1024.0
            elif im_type == UINT16:
                input_img = exp_img / 65535.0

            input_img = np.clip(input_img, 0.0, 1.0)
            input_img = input_img.astype(np.float16)

            if FLAG_APPROACH_PREDICT == 0:
                print("[INFO] predict image by orignal approach for {} block".format(i))
                result = core_orignal_predict(input_img, config.im_bands, model, config.window_size, config.img_w)
                result_mask[start:end,:]=result[:this_h,:]

            elif FLAG_APPROACH_PREDICT == 1:
                print("[INFO] predict image by smooth approach for {} block".format(i))
                output_mask = np.zeros((this_h+config.window_size, W), np.uint8)
                if out_bands > 1:
                    result = predict_img_with_smooth_windowing(
                        input_img,
                        model,
                        window_size=config.window_size,
                        subdivisions=config.subdivisions,
                        slices= config.slices,
                        real_classes=target_class,  # output channels = 是真的类别，总类别-背景
                        pred_func=core_smooth_predict_multiclass,
                        PLOT_PROGRESS=False
                    )
                    for i in range(target_class):
                        indx = np.where(result[:, :, i] >= 127)
                        output_mask[indx] = i + 1
                else:
                    result = predict_img_with_smooth_windowing(
                        input_img,
                        model,
                        window_size=config.window_size,
                        subdivisions=config.subdivisions,
                        slices=config.slices,
                        real_classes=target_class,
                        pred_func=core_smooth_predict_binary,
                        PLOT_PROGRESS=False
                    )
                    indx = np.where(result[:, :, 0] >= 127)
                    output_mask[indx] = 1

                result_mask[start:end, :] = output_mask[:this_h, :]

        gc.collect()

        print(np.unique(result_mask))
        result_mask[nodata_indx]=255
        output_file = ''.join([output_dir, '/', abs_filename, config.suffix])
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(output_file, W, H, 1, gdal.GDT_Byte)
        outdataset.SetGeoTransform(geoinf)
        if outdataset == None:
            print("create dataset failed!\n")
            sys.exit(-2)
        outdataset.GetRasterBand(1).WriteArray(result_mask)
        del outdataset
        # result_mask[nodata_indx] = 255
        gc.collect()
        print("Saved to:{}".format(output_file))

        # output vector file from raster file
        shp_file= ''.join([output_dir, '/', abs_filename, '.shp'])
        polygonize(output_file, shp_file)

        # cv2.imwrite(output_file, output_mask)




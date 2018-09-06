#coding:utf8
""""
    This is main procedure for remote sensing image semantic segmentation

"""
import cv2
import numpy as np
import os
import sys
import argparse
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras.preprocessing.image import img_to_array

from keras import backend as K
K.set_image_dim_ordering('tf')
K.clear_session()

from base_predict_functions import orignal_predict_onehot, smooth_predict_for_multiclass
from ulitities.base_functions import load_img_normalization_by_cv2, load_img_by_gdal, UINT10,UINT8,UINT16
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands

"""
   The following global variables should be put into meta data file 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


window_size = 256
step = 128

im_bands = 3
im_type = UINT8

dict_network={0: 'unet', 1: 'fcnnet', 2: 'segnet'}
dict_target={0: 'roads', 1: 'buildings'}
target_class=len(dict_target)

FLAG_USING_NETWORK = 0  # 0:unet; 1:fcn; 2:segnet;

FLAG_APPROACH_PREDICT=1  # 0: original predict, 1: smooth predict

img_file = '../../data/test/jian11_test_nrg.png'  # jian11_test_nrg, sample1_nrg

# model_file = ''.join(['../../data/models/sat_urban_nrg/',dict_network[FLAG_USING_NETWORK], '_multiclass.h5'])
model_file = '/home/omnisky/PycharmProjects/data/models/output/unet_multiclass_2018-09-06_10-54-12.h5'
# model_file = '/home/omnisky/PycharmProjects/data/models/sat_urban_nrg/unet_multiclass.h5'

if __name__ == '__main__':

    print("[INFO] opening image...")

    # ret, input_img = load_img_normalization_by_cv2(img_file)

    input_img = load_img_by_gdal(img_file)
    if im_type == UINT8:
        input_img = input_img / 255.0
    elif im_type == UINT10:
        input_img = input_img / 1024.0
    elif im_type == UINT16:
        input_img = input_img / 65535.0
    input_img = np.clip(input_img, 0.0, 1.0)


    abs_filename = os.path.split(img_file)[1]
    abs_filename = abs_filename.split(".")[0]
    print (abs_filename)

    """checke model file"""
    print("model file: {}".format(model_file))
    if not os.path.isfile(model_file):
        print("model does not exist:{}".format(model_file))
        sys.exit(-2)

    model= load_model(model_file)

    if FLAG_APPROACH_PREDICT==0:
        print("[INFO] predict image by orignal approach\n")
        result = orignal_predict_onehot(input_img, im_bands, model, window_size)
        output_file = ''.join(['../../data/predict/original_predict_',abs_filename, '.png'])
        print("result save as to: {}".format(output_file))
        cv2.imwrite(output_file, result*100)

    elif FLAG_APPROACH_PREDICT==1:
        print("[INFO] predict image by smooth approach\n")
        result = predict_img_with_smooth_windowing_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            subdivisions=2,
            real_classes=target_class,  # output channels = 是真的类别，总类别-背景
            pred_func=smooth_predict_for_multiclass
        )

        for b in range(target_class):
            output_file = ''.join(['../../data/predict/', dict_network[FLAG_USING_NETWORK], '/mask_multiclass_',
                                   abs_filename, '_', dict_target[b], '.png'])
            print("result save as to: {}".format(output_file))
            cv2.imwrite(output_file, result[:,:,b])



#coding:utf8
""""
    This is main procedure for remote sensing image semantic segmentation

"""
import cv2
import numpy as np
import os
import sys
import gc
import argparse
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras.preprocessing.image import img_to_array

from keras import backend as K
K.set_image_dim_ordering('tf')
K.clear_session()

from base_predict_functions import orignal_predict_notonehot, smooth_predict_for_binary_notonehot
from ulitities.base_functions import load_img_normalization_by_cv2, load_img_by_gdal, UINT10,UINT8,UINT16
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands
# from semantic_segmentation_networks import jaccard_coef,jaccard_coef_int

"""
   The following global variables should be put into meta data file 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


target_class =1

window_size = 256  #224, 256, 288. 320
# step = 128

im_bands =4
im_type = UINT10  # UINT10,UINT8,UINT16
dict_network={0: 'unet', 1: 'fcnnet', 2: 'segnet'}
dict_target={0: 'roads', 1: 'buildings'}
FLAG_USING_NETWORK = 0  # 0:unet; 1:fcn; 2:segnet;

FLAG_TARGET_CLASS = 0  # 0:roads; 1:buildings

FLAG_APPROACH_PREDICT = 1  # 0: original predict, 1: smooth predict

# position = 'shuangliu_1test' #  1)jian11_test, , 2)jiangyou, 3)yujiang_test,
# 4)cuiping, 5)shuangliu_1test, 6) tongchuan_test
# 7) lizhou_test, 8) jianyang, 9)yushui22_test, 10) sample1, 11)ruoergai_52test
# img_file = '../../data/test/paper/images/'+position+'_4bands1024.png'  # _rgb, _nrg, _4bands1024.
img_file = '/home/omnisky/PycharmProjects/data/test/ducha/cd13_test_src.png'
# img_file='/home/omnisky/PycharmProjects/data/test/sample1_12.png'


# model_file = ''.join(['../../data/models/sat_urban_rgb/',dict_network[FLAG_USING_NETWORK], '_',
#                       dict_target[FLAG_TARGET_CLASS],'_binary_jaccard_', str(window_size), '_final.h5'])
model_file ='/home/omnisky/PycharmProjects/data/models/ducha/tuitiantu_jaccardandCross_2018-12-29_09-14-05.h5'

print("model: {}".format(model_file))

if __name__ == '__main__':

    print("[INFO] opening image...")
    # ret, input_img = load_img_normalization_by_cv2(img_file)
    # if ret !=0:
    #     print("Open input file failed: {}".format(img_file))
    # sys.exit(-1)

    input_img = load_img_by_gdal(img_file)
    if im_type == UINT8:
        input_img = input_img / 255.0
    elif im_type == UINT10:
        input_img = input_img / 1024.0
    elif im_type == UINT16:
        input_img = input_img / 65535.0

    input_img = np.clip(input_img, 0.0, 1.0)
    input_img = input_img.astype(np.float16)  # test accuracy


    abs_filename = os.path.split(img_file)[1]
    abs_filename = abs_filename.split(".")[0]
    print (abs_filename)

    """checke model file"""
    print("model file: {}".format(model_file))
    if not os.path.isfile(model_file):
        print("model does not exist:{}".format(model_file))
        sys.exit(-2)

    model = load_model(model_file)

    if FLAG_APPROACH_PREDICT==0:
        print("[INFO] predict image by orignal approach\n")
        result = orignal_predict_notonehot(input_img,im_bands, model, window_size)
        # output_file = ''.join(['../../data/predict/',dict_network[FLAG_USING_NETWORK],'/sat_4bands/original_pred_',
        #                        abs_filename, '_', dict_target[FLAG_TARGET_CLASS],'_jaccard.png'])
        output_file = ''.join(['../../data/test/tianfuxinqu/pred/pred_', str(window_size), '/mask_binary_',
                               abs_filename, '_', dict_target[FLAG_TARGET_CLASS], '_jaccard_original.png'])
        output_file = '/home/omnisky/PycharmProjects/data/originaldata/zs/pred/b_pred_original.png'
        print("result save as to: {}".format(output_file))
        cv2.imwrite(output_file, result*128)

    elif FLAG_APPROACH_PREDICT==1:
        print("[INFO] predict image by smooth approach\n")
        result = predict_img_with_smooth_windowing_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            subdivisions=2,
            real_classes=target_class,  # output channels = 是真的类别，总类别-背景
            pred_func=smooth_predict_for_binary_notonehot
        )
        # output_file = ''.join(['../../data/predict/', dict_network[FLAG_USING_NETWORK],'/sat_rgb/mask_binary_',str(window_size),
        #                        '_', abs_filename, '_', dict_target[FLAG_TARGET_CLASS],'_jaccard.png'])
        # output_file = ''.join(['../../data/test/tianfuxinqu/pred/pred_', str(window_size), '/mask_binary_',
        #                        abs_filename, '_', dict_target[FLAG_TARGET_CLASS], '_jaccard_smooth.png'])
        output_file = '/home/omnisky/PycharmProjects/data/originaldata/zs/pred/b_pred_512.png'
        print("result save as to: {}".format(output_file))

        cv2.imwrite(output_file, result)
        print("Saved to {}".format(output_file))

    gc.collect()



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
#
from segnet_predict import predict, predict_for_segnet_multiclassbands,get_predicted_pathces_from_image, mosaic_resut,predict_for_segnet_grayresult
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands,cheap_tiling_prediction_not_square_img_multiclassbands

from unet_predict import unet_predict,predict_for_unet_multiclassbands

from keras import backend as K
K.set_image_dim_ordering('th')
# K.set_image_dim_ordering('tf')
"""
   The following global variables should be put into meta data file 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# segnet_classes = [0., 1., 2., 3., 4.]
segnet_classes = [0., 1., 2.]
segnet_dict = {'road':1, 'building':2}

unet_classes = [0., 1.]


FLAG_USING_UNET = True
test_image_path = '../data/test/3.png'
# model path & test_image path

# unet_model_path = '../data/models/unet_channel_first_roads.h5'
# unet_output_mask = '../data/predict/unet/mask_unet_roads_'+os.path.split(test_image_path)[1]

unet_model_path = '../data/models/unet_channel_first_buildings.h5'
unet_output_mask = '../data/predict/unet/mask_unet_buildings_'+os.path.split(test_image_path)[1]

segnet_model_path = '../data/models/segnet_channel_first_012labels.h5' # for channel_first
segnet_output_path = '../data/predict/segnet/mask_segnet_'


window_size = 256

step = 128


if __name__ == '__main__':

    print("[INFO] opening image...")
    input_img = cv2.imread(test_image_path)

    print("[INFO] loading network...")
    if FLAG_USING_UNET:
        model = load_model(unet_model_path)
        result_channels = len(unet_classes) - 1

        labelencoder = LabelEncoder()
        labelencoder.fit(unet_classes)

    else:
        model = load_model(segnet_model_path)

        result_channels = len(segnet_classes) - 1

        labelencoder = LabelEncoder()
        labelencoder.fit(segnet_classes)

    """1. test original code of predict()"""
    # if FLAG_USING_UNET:
    #     result_test=unet_predict(input_img, model, window_size, labelencoder)
    # else:
    #     result_test=predict(input_img, model, window_size,labelencoder)
    #
    # cv2.imwrite('../data/predict/test.png', result_test)
    # sys.exit()

    """2. test code of flame tracer """
    # predicted_patches = get_predicted_pathces_from_image(
    #     input_img,
    #     model,
    #     window_size,
    #     step,
    #     pre_func=predict_for_segnet_grayresult,
    #     labelencoder=labelencoder
    # )
    #
    # mosaic_resut(predicted_patches)
    # sys.exit()

    """ 3. true predict """

    """3.1 test cheap """
    # if FLAG_USING_UNET:
    #     predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
    #         input_img,
    #         model,
    #         window_size=window_size,
    #         real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
    #         pred_func=predict_for_unet_multiclassbands,
    #         labelencoder=labelencoder
    #     )
    #     cv2.imwrite(unet_output_mask, predictions_cheap)
    # else:
    #     predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
    #         input_img,
    #         model,
    #         window_size=window_size,
    #         real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
    #         pred_func=predict_for_segnet_multiclassbands,
    #         labelencoder=labelencoder
    #     )
    #     for key,val in segnet_dict.items():
    #         output_file = segnet_output_path+key+'.png'
    #         cv2.imwrite(output_file, predictions_cheap[:,:,val-1])  # achieve the integer automatically
    #
    #
    # sys.exit()

    if FLAG_USING_UNET:
        predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            subdivisions=2,
            real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
            pred_func=predict_for_unet_multiclassbands,
            labelencoder=labelencoder
        )
        cv2.imwrite(unet_output_mask, predictions_smooth)
    else:
        predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            subdivisions=2,
            real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
            pred_func=predict_for_segnet_multiclassbands,
            labelencoder=labelencoder
        )

        for key,val in segnet_dict.items():
            output_file = segnet_output_path+key+'.png'
            cv2.imwrite(output_file, predictions_smooth[:,:,val-1])  # achieve the integer automatically



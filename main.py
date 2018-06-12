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
#
#
from segnet_predict import predict, predict_for_segnet_multiclassbands,get_predicted_pathces_from_image, mosaic_resut,predict_for_segnet_grayresult
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands,cheap_tiling_prediction_not_square_img_multiclassbands


"""
   The following global variables should be put into meta data file 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

classes = [0., 1., 2., 3., 4.]
result_channels = len(classes) - 1

labelencoder = LabelEncoder()
labelencoder.fit(classes)

# model path & test_image path

model_path = './data/models/segnet.h5'
test_image_path = './data/test/industrial_3.png'

window_size = 256

step = 128

if __name__ == '__main__':

    print("[INFO] opening image...")
    input_img = cv2.imread(test_image_path)

    print("[INFO] loading network...")
    model = load_model(model_path)

    """1. test original code of predict()"""
    # predict(input_img, model, window_size,labelencoder)
    # sys.eixt()

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
    # sys.eixt()

    """ 3. true predict by segnet """

    """3.1 test cheap """

    # predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
    #     input_img,
    #     model,
    #     window_size=window_size,
    #     real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
    #     pred_func=predict_for_segnet_multiclassbands,
    #     labelencoder=labelencoder
    # )
    # cv2.imwrite('./data/predict/pre_cheap_multibands.png', predictions_cheap)
    #
    # sys.exit()

    predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
        input_img,
        model,
        window_size=window_size,
        subdivisions=2,
        real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
        pred_func=predict_for_segnet_multiclassbands,
        labelencoder=labelencoder
    )
    cv2.imwrite('./data/predict/predictions_smooth_multiclasses.png', predictions_smooth)


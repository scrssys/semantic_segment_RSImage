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

segnet_classes = [0., 1., 2., 3., 4.]
unet_classes = [0., 1.]


# model path & test_image path

# unet_model_path = '../data/models/unet_channel_firstbuldlings.h5'
unet_model_path = '../data/models/unet_channel_first_buildings.h5'
# segnet_model_path = '../data/models/segnet_train_test_1.h5'
# segnet_model_path = '../data/models/segnet_channel_last.h5'
segnet_model_path = '../data/models/segnet_channel_first.h5' # for channel_first
test_image_path = '../data/test/1.png'
output_mask = '../data/predict/unet/mask_buildings_img_'+os.path.split(test_image_path)[1]

window_size = 256

step = 128

FLAG_USING_UNET = True



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
    #     unet_predict(input_img, model, window_size, labelencoder)
    # else:
    #     predict(input_img, model, window_size,labelencoder)
    #
    #
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
    if FLAG_USING_UNET:
        predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
            pred_func=predict_for_unet_multiclassbands,
            labelencoder=labelencoder
        )
    else:
        predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
            pred_func=predict_for_segnet_multiclassbands,
            labelencoder=labelencoder
        )
    # cv2.imwrite('./data/predict/pre_cheap_multibands.png', predictions_cheap)
    cv2.imwrite(output_mask, predictions_cheap)

    # sys.exit()

    # if FLAG_USING_UNET:
    #     predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
    #         input_img,
    #         model,
    #         window_size=window_size,
    #         subdivisions=2,
    #         real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
    #         pred_func=predict_for_unet_multiclassbands,
    #         labelencoder=labelencoder
    #     )
    # else:
    #     predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
    #         input_img,
    #         model,
    #         window_size=window_size,
    #         subdivisions=2,
    #         real_classes=result_channels,  # output channels = 真是的类别，总类别-背景
    #         pred_func=predict_for_segnet_multiclassbands,
    #         labelencoder=labelencoder
    #     )

    # cv2.imwrite(output_mask, predictions_smooth)


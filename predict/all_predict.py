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
from segnet_predict import segnet_predict_binary, segnet_predict_multiclass, predict_for_segnet_multiclassbands,get_predicted_pathces_from_image, mosaic_resut,predict_for_segnet_grayresult
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands,cheap_tiling_prediction_not_square_img_multiclassbands

from unet_predict import unet_predict_binary, unet_predict_multiclass, smooth_predict_for_unet_binary,smooth_predict_for_unet_multiclass


from keras import backend as K
K.set_image_dim_ordering('th')
# K.set_image_dim_ordering('tf')
"""
   The following global variables should be put into meta data file 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

multi_classes = [0., 1., 2.]
multi_dict = {'road':1, 'building':2}


binary_classes = [0., 1.]


# FLAG_USING_UNET = True
# FLAG_USING_UNET = False
FLAG_APPROACH_PREDICT=3 # 0: raw predict; 1:flame tracer for smooth; 2: cheap predict; else:smooth predict
FLAG_USING_MODEL=1 # 0:Unet two-category; 1:Unet multi-category; 2:segnet two-category; 3:segnet multi-category


input_image = '../../data/test/3.png'

#
# """(1.1) for unet road predict"""
# # unet_model_path = '../../data/models/unet_channel_first_roads.h5'
# # unet_output_mask = '../../data/predict/unet/mask_unet_roads_'+os.path.split(test_image_path)[1]
#
# """(1.2)for unet road predict"""
# unet_model_path = '../../data/models/unet_channel_first_buildings.h5'
# unet_output_mask = '../../data/predict/unet/mask_unet_buildings_'+os.path.split(input_image)[1]
#
#
# """(2) for segnet predict"""
# segnet_model_path = '../../data/models/segnet_channel_first.h5' # for channel_first
# segnet_output_path = '../../data/predict/segnet/mask_segnet_new_'

window_size = 256

step = 128


if __name__ == '__main__':

    print("[INFO] opening image...")
    if not os.path.isfile(input_image):
        print("no file: {}".forma(input_image))
        sys.exit(-1)

    input_img = cv2.imread(input_image)
    input_img = np.array(input_img, dtype="float") / 255.0  # must do it

    print("[INFO] loading network...")
    if FLAG_USING_MODEL==0:
        model = load_model('../../data/models/unet_channel_first_buildings.h5')  # for buildings
        # model = load_model('../../data/models/unet_channel_first_roads.h5') # for roads
        result_channels = len(binary_classes) - 1
        output_mask = '../../data/predict/unet/mask_binary_buildings_'+os.path.split(input_image)[1]
        # labelencoder = LabelEncoder()
        # labelencoder.fit(unet_classes)
    elif FLAG_USING_MODEL==1:
        model = load_model('../../data/models/unet_channel_first_multiclass.h5')
        result_channels = len(multi_classes)-1
        output_mask = '../../data/predict/unet/mask_multiclass_'
    elif FLAG_USING_MODEL==2:
        model = load_model('../../data/models/segnet_channel_first_buildings.h5')  # for buildings
        # model = load_model('../../data/models/segnet_channel_first_roads.h5') # for roads
        result_channels = len(binary_classes) - 1
        output_mask = '../../data/predict/segnet/mask_binary_buildings_'+os.path.split(input_image)[1]
        # labelencoder = LabelEncoder()
        # labelencoder.fit(unet_classes)
    elif FLAG_USING_MODEL==3:
        model = load_model('../../data/models/segnet_channel_first_multiclass.h5')
        result_channels = len(multi_classes)-1
        output_mask = '../../data/predict/segnet/mask_multiclass_'


    if FLAG_APPROACH_PREDICT ==0:
        """0. test original code of predict()"""
        if FLAG_USING_MODEL==0:
            result_test = unet_predict_binary(input_img, model, window_size)
            cv2.imwrite(output_mask, result_test)
        if FLAG_USING_MODEL == 1:
            result_test = unet_predict_multiclass(input_img, model, window_size, result_channels)
            for key, val in multi_dict.items():
                output_file = output_mask + key + '.png'
                cv2.imwrite(output_file, result_test[:, :, val - 1])  # achieve the integer automatically
        elif FLAG_USING_MODEL == 2:
            result_test = segnet_predict_binary(input_img, model, window_size)
            cv2.imwrite(output_mask, result_test)
        elif FLAG_USING_MODEL == 3:
            result_test = segnet_predict_multiclass(input_img, model, window_size, result_channels)
            for key, val in multi_dict.items():
                output_file = output_mask + key + '.png'
                cv2.imwrite(output_file, result_test[:, :, val - 1])  # achieve the integer automatically

    if FLAG_APPROACH_PREDICT ==3:
        """0. test original code of predict()"""
        if FLAG_USING_MODEL==0:
            predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
                input_img,
                model,
                window_size=window_size,
                subdivisions=2,
                real_classes=result_channels,  # output channels = 是真的类别，总类别-背景
                pred_func=smooth_predict_for_unet_binary
                # labelencoder=labelencoder
            )

            cv2.imwrite(output_mask, predictions_smooth)
        if FLAG_USING_MODEL == 1:
            predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
                input_img,
                model,
                window_size=window_size,
                subdivisions=2,
                real_classes=result_channels,  # output channels = 是真的类别，总类别-背景
                pred_func=smooth_predict_for_unet_multiclass
                # labelencoder=labelencoder
            )
            for key, val in multi_dict.items():
                output_file = output_mask + key + '.png'
                cv2.imwrite(output_file, predictions_smooth[:, :, val - 1])  # achieve the integer automatically
        # not complete
        # elif FLAG_USING_MODEL == 2:
        #     result_test = segnet_predict_binary(input_img, model, window_size)
        #     cv2.imwrite(output_mask, result_test)
        # elif FLAG_USING_MODEL == 3:
        #     result_test = segnet_predict_multiclass(input_img, model, window_size, result_channels)
        #     for key, val in multi_dict.items():
        #         output_file = output_mask + key + '.png'
        #         cv2.imwrite(output_file, result_test[:, :, val - 1])  # achieve the integer automatically


    # elif FLAG_APPROACH_PREDICT==1:
    #     """1. test code of flame tracer """
    #     predicted_patches = get_predicted_pathces_from_image(
    #         input_img,
    #         model,
    #         window_size,
    #         step,
    #         pre_func=predict_for_segnet_grayresult,
    #         labelencoder=labelencoder)
    #     mosaic_resut(predicted_patches)
    #
    # elif FLAG_APPROACH_PREDICT==2:
    #     """2. test cheap  predict"""
    #     if FLAG_USING_UNET:
    #         predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
    #             input_img,
    #             model,
    #             window_size=window_size,
    #             real_classes=result_channels, # output channels = 是真的类别，总类别-背景
    #             pred_func=predict_for_unet_multiclassbands,
    #             labelencoder=labelencoder
    #         )
    #         cv2.imwrite(unet_output_mask, predictions_cheap)
    #     else:
    #         predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
    #             input_img,
    #             model,
    #             window_size=window_size,
    #             real_classes=result_channels,  # output channels = 是真的类别，总类别-背景
    #             pred_func=predict_for_segnet_multiclassbands,
    #             labelencoder=labelencoder
    #         )
    #         for key,val in segnet_dict.items():
    #             output_file = segnet_output_path+key+'.png'
    #             cv2.imwrite(output_file, predictions_cheap[:,:,val-1])  # achieve the integer automatically
    #
    # else:
    #     """3. smooth predict """
    #     if FLAG_USING_UNET:
    #         predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
    #             input_img,
    #             model,
    #             window_size=window_size,
    #             subdivisions=2,
    #             real_classes=result_channels,  # output channels = 是真的类别，总类别-背景
    #             pred_func=predict_for_unet_multiclassbands,
    #             labelencoder=labelencoder
    #         )
    #         cv2.imwrite(unet_output_mask, predictions_smooth)
    #     else:
    #         predictions_smooth = predict_img_with_smooth_windowing_multiclassbands(
    #             input_img,
    #             model,
    #             window_size=window_size,
    #             subdivisions=2,
    #             real_classes=result_channels, # output channels = 是真的类别，总类别-背景
    #             pred_func=predict_for_segnet_multiclassbands,
    #             labelencoder=labelencoder
    #         )
    #         for key,val in segnet_dict.items():
    #             output_file = segnet_output_path+key+'.png'  # try join([])
    #             cv2.imwrite(output_file, predictions_smooth[:,:,val-1])
    #             # imwrite () could achieve the int format  automatically



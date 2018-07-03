#coding:utf-8

import cv2
import random
import numpy as np
import os
import sys
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

import gc
from smooth_tiled_predictions import predict_img_with_smooth_windowing, cheap_tiling_prediction_not_square_img,cheap_tiling_prediction_not_square_img_multiclassbands, predict_img_with_smooth_windowing_multiclassbands

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th') # channel_first
# K.set_image_dim_ordering('tf')

import mxnet as mx


def unet_predict(image, model, window_size, labelencoder):

    stride = window_size

    h, w, _ = image.shape
    print 'h,w:', h, w
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = image[:, :, :]
    padding_img = padding_img.astype("float") / 255.0

    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):

            crop = padding_img[:3, i * stride:i * stride + window_size, j * stride:j * stride + window_size]
            # crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]

            cb, ch, cw = crop.shape
            print ('crop:{}'.format(crop.shape))
            print (np.unique(crop))

            crop = np.expand_dims(crop, axis=0)
            print ('crop:{}'.format(crop.shape))


            pred = model.predict(crop, verbose=2)
            # print (np.unique(pred))
            pred = pred +0.5

            pred = pred.reshape((256, 256)).astype(np.uint8)
            print (np.unique(pred))

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult = mask_whole[0:h, 0:w]

    plt.imshow(outputresult,cmap='gray')
    plt.title("Original predicted result")
    plt.show()

    cv2.imwrite('../data/predict/testorignalpredict.png', outputresult)

def predict_for_unet_multiclassbands(small_img_patches, model, real_classes,labelencoder):
    """
        Apply prediction on images arranged in a 4D array as a batch.(patches, row,column, channels)
        output is a (pathes, x, y, real_classes): a multiband image
    """
    small_img_patches = np.array(small_img_patches)
    print (small_img_patches.shape)
    assert (len(small_img_patches.shape) == 4)

    patches,row,column,input_channels = small_img_patches.shape
    # assert (input_channels < row)

    mask_output = []
    for p in range(patches):
        crop = np.zeros((row, column, input_channels), np.uint8)
        crop = small_img_patches[p,:,:,:]
        crop = crop / 255.0
        # crop = crop.transpose(2,0,1) # for channel_first
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict(crop, verbose=2)
        pred += 0.5

        pred = pred.reshape((row,column)).astype(np.uint8)
        # 将预测结果分波段存储
        res_pred = np.zeros((row,column,real_classes), np.uint8)
        for t in range(real_classes):
            for i in range(row):
                for j in range(column):
                    if pred[i,j] ==t+1:
                        res_pred[i,j,t]=1

        mask_output.append(res_pred)

    mask_output = np.array(mask_output)
    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output

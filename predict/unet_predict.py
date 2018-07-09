#coding:utf-8

import cv2
import random
import numpy as np
import os
import sys
import argparse
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th') # channel_first
# K.set_image_dim_ordering('tf')


def unet_predict(image, model, window_size, labelencoder):

    stride = window_size

    h, w, _ = image.shape
    print 'h,w:', h, w
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3))
    padding_img[0:h, 0:w, :] = image[:, :, :]

    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            crop = padding_img[:3, i * stride:i * stride + window_size, j * stride:j * stride + window_size]
            # crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]

            crop = np.expand_dims(crop, axis=0)
            print ('crop:{}'.format(crop.shape))

            pred = model.predict(crop, verbose=2)

            pred = pred.reshape(256, 256)
            print (np.unique(pred))

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult = mask_whole[0:h, 0:w]*255.0

    plt.imshow(outputresult,cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    return outputresult


def predict_for_unet_multiclassbands(small_img_patches, model, real_classes,labelencoder):
    """

    :param small_img_patches: input image 4D array (patches, row,column, channels)
    :param model: pretrained model
    :param real_classes: the number of classes and the channels of output mask
    :param labelencoder:
    :return: predict mask 4D array (patches, row,column, real_classes)
    """

    assert(real_classes ==1 ) # only usefully for one class

    small_img_patches = np.array(small_img_patches)
    print (small_img_patches.shape)
    assert (len(small_img_patches.shape) == 4)

    patches,row,column,input_channels = small_img_patches.shape

    mask_output = []
    for p in range(patches):
        # crop = np.zeros((row, column, input_channels), np.uint8)
        crop = small_img_patches[p,:,:,:]
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict(crop, verbose=2)
        pred = pred.reshape((row,column))

        # 将预测结果2D expand to 3D
        res_pred = np.expand_dims(pred, axis=-1)

        mask_output.append(res_pred)

    mask_output = np.array(mask_output)
    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output

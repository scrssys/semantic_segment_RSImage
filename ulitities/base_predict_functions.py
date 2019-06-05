#coding:utf-8

import cv2
import random
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import gc
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

from keras import backend as K
# K.set_image_dim_ordering('th') # channel_first
K.set_image_dim_ordering('tf')


def smooth_predict_for_binary_onehot(small_img_patches, model, real_classes):
    """

    :param small_img_patches: input image 4D array (patches, row,column, channels)
    :param model: pretrained model
    :param real_classes: the number of classes and the channels of output mask
    :param labelencoder:
    :return: predict mask 4D array (patches, row,column, real_classes)
    """

    assert(real_classes ==1)  # only usefully for binary classification

    small_img_patches = np.array(small_img_patches)
    print (small_img_patches.shape)
    assert (len(small_img_patches.shape) == 4)

    patches,row,column,input_channels = small_img_patches.shape

    mask_output = []
    for p in list(range(patches)):
        crop = small_img_patches[p,:,:,:]
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict(crop, verbose=2)
        pred = np.argmax(pred, axis=2)
        # print(np.unique(pred))
        pred = pred.reshape((row,column))

        # 将预测结果2D expand to 3D
        res_pred = np.expand_dims(pred, axis=-1)

        mask_output.append(res_pred)

    mask_output = np.array(mask_output,np.float16)
    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output

def smooth_predict_for_binary_notonehot(small_img_patches, model, real_classes):
    """

    :param small_img_patches: input image 4D array (patches, row,column, channels)
    :param model: pretrained model
    :param real_classes: the number of classes and the channels of output mask
    :param labelencoder:
    :return: predict mask 4D array (patches, row,column, real_classes)
    """

    assert(real_classes ==1)  # only usefully for binary classification
    # small_img_patches = small_img_patches.astype(np.float32)

    small_img_patches = np.array(small_img_patches)
    print (small_img_patches.shape)
    assert (len(small_img_patches.shape) == 4)

    patches,row,column,input_channels = small_img_patches.shape

    mask_output = []
    for p in list(range(patches)):
        crop = small_img_patches[p,:,:,:]
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict(crop, verbose=2)
        # pred = np.argmax(pred, axis=2)
        # print(np.unique(pred))
        pred[pred<0.5]=0
        pred[pred>=0.5]=1
        pred = pred.reshape((row,column))

        # 将预测结果2D expand to 3D
        res_pred = np.expand_dims(pred, axis=-1)

        mask_output.append(res_pred)

    mask_result = np.array(mask_output, np.float16)
    del mask_output, small_img_patches
    gc.collect()

    print ("Shape of mask_output:{}".format(mask_result.shape))

    return mask_result

def smooth_predict_for_multiclass(small_img_patches, model, real_classes):
    """

    :param small_img_patches: input image 4D array (patches, row,column, channels)
    :param model: pretrained model
    :param real_classes: the number of classes and the channels of output mask
    :param labelencoder:
    :return: predict mask 4D array (patches, row,column, real_classes)
    """

    small_img_patches = np.array(small_img_patches)
    print (small_img_patches.shape)
    assert (len(small_img_patches.shape) == 4)

    patches,row,column,input_channels = small_img_patches.shape

    mask_output = []
    for p in list(range(patches)):
        crop = small_img_patches[p,:,:,:]
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict(crop, verbose=2)
        if len(pred.shape) > 3:
            pred = np.argmax(pred, axis=3)
        else:
            pred = np.argmax(pred, axis=2)

        # pred = np.argmax(pred, axis=2)
        pred = pred.reshape((row*column))
        # mask_output.append(pred)

        """using index function "where" to rapid find different class"""
        tmp = pred.astype(np.uint8)
        res_pred = np.zeros((row * column, real_classes))
        for t in list(range(real_classes)):
            idx = np.where(tmp == t + 1)
            res_pred[idx, t] = 1
        res_pred = res_pred.reshape((row, column, real_classes))

        """bad demo as following: (cost long time by for loop)"""
        # """method 2: by looping through every index"""
        # res_pred = np.zeros((row, column, real_classes))
        # for i in range(row):
        #     for j in range(column):
        #         for t in range(real_classes):
        #             if pred[i,j]==t+1:
        #                 res_pred[i,j,t]=1

        mask_output.append(res_pred)

    mask_output = np.array(mask_output, np.float16)
    print(np.unique(mask_output))

    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output


# window_size=256

def orignal_predict_onehot(image,bands, model,window_size):
    stride = window_size

    h, w, _ = image.shape
    print('h,w:', h, w)
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, bands))
    padding_img[0:h, 0:w, :] = image[:, :, :]

    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in list(range(padding_h // stride)):
        for j in list(range(padding_w // stride)):
            crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :bands]

            crop = np.expand_dims(crop, axis=0)
            # print('crop:{}'.format(crop.shape))

            pred = model.predict(crop, verbose=2)
            if len(pred.shape) > 3:
                pred = np.argmax(pred, axis=3)
            else:
                pred = np.argmax(pred, axis=2)

            # pred = np.argmax(pred, axis=2)  #for one hot encoding
            # pred = pred[:,:,1]

            pred = pred.reshape(256, 256)
            print(np.unique(pred))

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult =mask_whole[0:h,0:w]
    # outputresult = outputresult.astype(np.uint8)
    # outputresult[outputresult>0.8]=255
    # outputresult[outputresult <= 0.5] = 0

    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    # cv2.imwrite('../../data/predict/test_model.png',outputresult*255)
    return outputresult

def orignal_predict_notonehot(image,bands, model,window_size):
    stride = window_size

    h, w, _ = image.shape
    print('h,w:', h, w)
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, bands))
    padding_img[0:h, 0:w, :] = image[:, :, :]

    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in tqdm(list(range(padding_h // stride))):
        for j in list(range(padding_w // stride)):
            crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :bands]

            crop = np.expand_dims(crop, axis=0)
            # print('crop:{}'.format(crop.shape))

            pred = model.predict(crop, verbose=2)
            # pred = np.argmax(pred, axis=2)  #for one hot encoding
            # pred = pred[:,:,1]
            pred[pred<0.5]=0
            pred[pred>=0.5]=1

            pred = pred.reshape(256, 256)
            # print(np.unique(pred))

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult =mask_whole[0:h,0:w]
    # outputresult = outputresult.astype(np.uint8)

    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    # cv2.imwrite('../../data/predict/test_model.png',outputresult*255)
    return outputresult
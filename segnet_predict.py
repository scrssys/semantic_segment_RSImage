# coding:utf-8

import cv2
import random
import numpy as np
import os
import sys
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import gc
from smooth_tiled_predictions import predict_img_with_smooth_windowing, cheap_tiling_prediction_not_square_img,cheap_tiling_prediction_not_square_img_multiclassbands, predict_img_with_smooth_windowing_multiclassbands

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array

from PIL import Image

from keras import backend as K
K.set_image_dim_ordering('th')  # for channel_first
# K.set_image_dim_ordering('tf')

# img_w = 256
#
# img_h = 256
# n_label =5


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
                    help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())
    return args


# test for predict_segnet()
def get_predicted_pathces_from_image(input_img, model, window_size, step, pre_func, labelencoder):

    # input_img = cv2.imread(img_path)
    input_img = np.array(input_img)
    # input_img = np.transpose(input_img,(2,0,1))
    print (input_img.shape)
    row, column, channel = input_img.shape
    assert channel < row
    subdivs = []
    for i in range(0,row-window_size+1, step):
        subdivs.append([])
        for j in range (0, column-window_size+1, step):
            patch = input_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)


    gc.collect()

    subdivs = np.array(subdivs)
    print ("shape of subdivs: {}".format(subdivs.shape))
    a,b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a*b, c, d, e)
    print ("shape of subdivs: {}".format(subdivs.shape))



    out_mask = pre_func(subdivs, model, window_size, labelencoder)
    print ("shape of out_mask:{}".format(out_mask.shape))
    # output image channel = 1
    out_img = out_mask.reshape(a,b,c,d,1)

    return out_img



def mosaic_resut(predicted_patches):
    '''

    :param predicted_patches: should be 4D np array (patch_row, patch_column, x, y, channels)
    :return: mosaic image (patch_row*x, patch_column*y, channels)
    '''

    assert (len(predicted_patches.shape) == 5)
    patch_row, patch_column, x, y, channels= predicted_patches.shape

    a =0
    result = np.zeros((patch_row*x, patch_column*y, channels), np.uint8)
    for i in range(0, patch_row*x, x):
        b=0
        for j in range(0, patch_column*y, y):
            result[i:i+x, j:j+y] = predicted_patches[a,b,:,:]
            b +=1
        a +=1

    cv2.imwrite('../data/predict/mosaic.png', result)






def predict_for_segnet_grayresult(small_img_patches, model, window_size,labelencoder):
    """
        Apply prediction on images arranged in a 4D array as a batch.
        output is a (pathes, x, y, 1): gray image
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
        crop = crop.transpose(2,0,1)
        crop = np.expand_dims(crop, axis=0)
        # print ('crop:{}'.format(crop.shape))
        pred = model.predict_classes(crop, verbose=2)
        pred = labelencoder.inverse_transform(pred[0])
        # print (np.unique(pred))
        pred = pred.reshape((window_size, window_size)).astype(np.uint8)
        pred = np.expand_dims(pred, axis=-1)
        mask_output.append(pred)

    mask_output = np.array(mask_output)
    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output


def predict_for_segnet_multiclassbands(small_img_patches, model, real_classes,labelencoder):
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

        # Using "img_to_array" to convert the dimensions ordering, to adapt "K.set_image_dim_ordering('**') "
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        pred = model.predict_classes(crop, verbose=2)
        pred = labelencoder.inverse_transform(pred[0])

        pred = pred.reshape((row,column)).astype(np.uint8)
        # 将预测结果分波段存储
        res_pred = np.zeros((row,column,real_classes), np.uint8)
        for t in range(real_classes):
            for i in range(row):
                for j in range(column):
                    if pred[i,j] ==t+1:
                        res_pred[i,j,t]=1
            # print("\nband:{}   labels contain:{}".format(t,np.unique(res_pred[:,:,t])))

        mask_output.append(res_pred)

    mask_output = np.array(mask_output)
    print ("Shape of mask_output:{}".format(mask_output.shape))

    return mask_output


def predict(image, model, window_size, labelencoder):

    stride = window_size

    h, w, _ = image.shape
    print 'h,w:', h, w
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = image[:, :, :]
    padding_img = padding_img.astype("float") / 255.0

    # Using "img_to_array" to convert the dimensions ordering, to adapt "K.set_image_dim_ordering('**') "
    padding_img = img_to_array(padding_img)
    print 'src:', padding_img.shape
    # padding_img = padding_img.transpose(2, 0, 1) # for channel_first

    print 'newsrc:', padding_img.shape
    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            crop = padding_img[:3, i * stride:i * stride + window_size, j * stride:j * stride + window_size]
            # crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]
            cb, ch, cw = crop.shape # for channel_first

            print ('crop:{}'.format(crop.shape))

            crop = np.expand_dims(crop, axis=0)
            # crop = crop.reshape((1,ch,cw,-1))
            print ('crop:{}'.format(crop.shape))
            pred = model.predict_classes(crop, verbose=2)
            pred = labelencoder.inverse_transform(pred[0])

            pred = pred.reshape(256, 256)
            # pred = np.swapaxes(pred,0,1)

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult = mask_whole[0:h, 0:w]*255.0
    plt.imshow(outputresult,cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    return outputresult
    # cv2.imwrite('../data/predict/pre1.png', mask_whole[0:h, 0:w])



def load_img(path, grayscale=False, target_size=None):
    img = Image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
    return img






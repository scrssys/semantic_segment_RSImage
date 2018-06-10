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
from smooth_tiled_predictions import predict_img_with_smooth_windowing, cheap_tiling_prediction_not_square_img,cheap_tiling_prediction_not_square_img_multiclassbands

import matplotlib.pyplot as plt

# sys.exit()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TEST_SET = ['1.png','2.png','3.png']
TEST_SET = ['1.png']
# TEST_SET = ['industrial_3.png']

image_size = 256

classes = [0., 1., 2., 3., 4.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
                    help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())
    return args


# model path & test_image path

model_path = './data/models/segnet.h5'
test_image_path = './data/test/industrial_3.png'

# import model
model = load_model(model_path)

window_size = 256

step = 128

# nb_classes = 5

# input_img = cv2.imread(test_image_path)


# test for predict_segnet()
def get_predicted_pathces_from_image(img_path):

    input_img = cv2.imread(img_path)
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



    out_mask = predict_for_segnet(subdivs)
    print ("shape of out_mask:{}".format(out_mask.shape))
    # output image channel = 1
    out_img = out_mask.reshape(a,b,c,d,1)

    return out_img

    # cv2.imwrite('./data/predict/pre_test.png', mask_output)
    # return subdivs


def mosaic_resut(predicted_patches):
    '''

    :param predicted_patches: should be 4D np array (patch_row, patch_column, x, y, channels)
    :return:
    '''

    assert (len(predicted_patches.shape) == 5)
    patch_row, patch_column, x, y, channels= predicted_patches.shape

    a =0
    result = np.zeros((patch_row*x, patch_column*y), np.uint8)
    for i in range(0, patch_row*x, x):
        b=0
        for j in range(0, patch_column*y, y):
            result[i:i+x, j:j+y] = predicted_patches[a,b,:,:,0]
            b +=1
        a +=1

    cv2.imwrite('./data/predict/pre_test.png', result)






def predict_for_segnet(small_img_patches):
    """
        Apply prediction on images arranged in a 4D array as a batch.
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


def predict_for_segnet_multiclassbands(small_img_patches, real_classes):
    """
        Apply prediction on images arranged in a 4D array as a batch.(patches, row,column, real_classes)
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





def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")

    model = load_model(model_path)
    # stride = args['stride']
    stride = image_size
    # for n in range(len(TEST_SET)):
        # path = TEST_SET[n]
        # load the image
        # image = cv2.imread('./test/' + path)
    image = cv2.imread(test_image_path)
    h, w, _ = image.shape
    print 'h,w:', h, w
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = image[:, :, :]
    padding_img = padding_img.astype("float") / 255.0
    padding_img = img_to_array(padding_img)
    print 'src:', padding_img.shape
    padding_img = padding_img.transpose(2, 0, 1)
    print 'newsrc:', padding_img.shape
    mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            crop = padding_img[:3, i * stride:i * stride + image_size, j * stride:j * stride + image_size]
            # crop = crop.transpose(crop,(2,0,1))
            cb, ch, cw = crop.shape
            print ('crop:{}'.format(crop.shape))
            # print 'cb,ch,cw:',cb,ch,cw
            # sys.exit(0)
            assert (ch == cw == 256)
            # if ch != 256 or cw != 256:
            #     print 'invalid size!'
            #     continue
                # sys.exit(0)

            crop = np.expand_dims(crop, axis=0)
            print ('crop:{}'.format(crop.shape))
            pred = model.predict_classes(crop, verbose=2)
            pred = labelencoder.inverse_transform(pred[0])
            # print (np.unique(pred))
            pred = pred.reshape((256, 256)).astype(np.uint8)
            # print 'pred:',pred.shape
            mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]
    plt.imshow(mask_whole,cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    cv2.imwrite('./data/predict/pre1.png', mask_whole[0:h, 0:w])


if __name__ == '__main__':
    # args = args_parse()

    """1. test original code of predict()"""
    # predict()
    # sys.eixt()


    """2. test code of flame tracer """
    # predicted_patches = get_predicted_pathces_from_image(test_image_path)

    # mosaic_resut(predicted_patches)

    """ 3. true predict by segnet """

    input_img = cv2.imread(test_image_path)

    """3.1 test cheap """
    real_classes = len(classes)-1
    predictions_cheap = cheap_tiling_prediction_not_square_img_multiclassbands(
        input_img,
        window_size=window_size,
        real_classes= real_classes,  # output channels = 真是的类别，总类别-背景
        pred_func=predict_for_segnet_multiclassbands
    )
    cv2.imwrite('./data/predict/pre_cheap_fore3bands.png', predictions_cheap)

    sys.exit()

    """3.2 test smooth """

    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size = window_size,
        subdivisions=2,
        nb_classes = 1, # output channels = 1
        pred_func = predict_for_segnet
    )
    cv2.imwrite('./data/predict/predictions_smooth.png', predictions_smooth)







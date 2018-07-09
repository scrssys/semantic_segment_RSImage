

import cv2
import numpy as np
import os
import sys
import argparse
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# segnet_classes = [0., 1., 2., 3., 4.]
unet_classes = [0., 1., 2.]

labelencoder = LabelEncoder()
labelencoder.fit(unet_classes)

input_image = '../../data/test/1.png'


"""(1.1) for unet road predict"""
unet_model_path = '../../data/models/unet_channel_first_multiclassnew.h5'
unet_output_mask = '../../data/predict/unet/mask_unet_roads_'+os.path.split(input_image)[1]


window_size = 256


if __name__=='__main__':
    input_img = cv2.imread(input_image)
    input_img = np.array(input_img, dtype="float") / 255.0  # must do it
    model = load_model(unet_model_path)

    stride = window_size

    h, w, _ = input_img.shape
    print 'h,w:', h, w
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3))
    padding_img[0:h, 0:w, :] = input_img[:, :, :]

    # Using "img_to_array" to convert the dimensions ordering, to adapt "K.set_image_dim_ordering('**') "
    padding_img = img_to_array(padding_img)
    print 'src:', padding_img.shape

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            crop = padding_img[:3, i * stride:i * stride + window_size, j * stride:j * stride + window_size]
            # crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]
            cb, ch, cw = crop.shape  # for channel_first

            print ('crop:{}'.format(crop.shape))

            crop = np.expand_dims(crop, axis=0)
            print ('crop:{}'.format(crop.shape))
            pred = model.predict(crop, verbose=2)
            # pred = pred.reshape()
            # pred = labelencoder.inverse_transform(pred[0])
            print(np.unique(pred))

            pred = pred[0,2,:]
            print(np.unique(pred))

            pred = pred.reshape(256, 256)
            # for ti in range(window_size):
            #     for tj in range(window_size):
            #         if pred[ti,tj]>0.01:
            #             pred[ti,tj]=1
            #         else:
            #             pred[ti,tj]=0

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult = mask_whole[0:h, 0:w] * 255.0
    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()




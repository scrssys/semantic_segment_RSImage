from keras.models import load_model
import os
import segmentation_models
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#
# modelfile='/home/omnisky/PycharmProjects/data/models/GF2/GF2_unet_vgg16_categorical_crossentropy_480_2019-06-20_08-26-16test.h5'
#
# model=load_model(modelfile)
# print(model.summary())

import numpy as np

A=np.zeros((256,250), np.int8)
print("shape of A:{}".format(A.shape))

B = np.expand_dims(A,axis=-1)
print("shape of B:{}".format(B.shape))
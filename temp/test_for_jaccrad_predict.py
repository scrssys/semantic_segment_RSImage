#coding=utf-8
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, History,ReduceLROnPlateau
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import sys
import os
from tqdm import tqdm
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras import backend as K
K.set_image_dim_ordering('tf')


from semantic_segmentation_networks import binary_unet_jaccard, binary_fcnnet_jaccard, binary_segnet_jaccard
from ulitities.base_functions import load_img_normalization

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256

n_label = 1

model_save_path='/home/omnisky/PycharmProjects/data/models/sat_urban_rgb/unet_buildings_binary_jaccard.h5'


window_size=256

def test_predict(image,model):
    stride = window_size

    h, w, _ = image.shape
    print('h,w:', h, w)
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img = np.zeros((padding_h, padding_w, 3))
    padding_img[0:h, 0:w, :] = image[:, :, :]

    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float32)
    for i in list(range(padding_h // stride)):
        for j in list(range(padding_w // stride)):
            crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]

            crop = np.expand_dims(crop, axis=0)
            print('crop:{}'.format(crop.shape))

            # pred = model.predict(crop, verbose=2)
            pred = model.predict(crop, verbose=2)
            # pred = np.argmax(pred, axis=2)  #for one hot encoding

            pred = pred.reshape(256, 256)
            # pred = pred[0]
            # pred = pred[:,:,0]
            print(np.unique(pred))


            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult =mask_whole[0:h,0:w]
    # outputresult = outputresult.astype(np.uint8)

    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    cv2.imwrite('../../data/predict/test_model.png',outputresult*255)
    return outputresult




if __name__ == '__main__':

    print("test ....................predict by trained model .....\n")
    test_img_path = '../../data/test/sample1.png'
    import sys

    if not os.path.isfile(test_img_path):
        print("no file: {}".format(test_img_path))
        sys.exit(-1)

    ret, input_img = load_img_normalization(test_img_path)
    # model_save_path ='../../data/models/unet_buildings_onehot.h5'

    new_model = load_model(model_save_path)

    test_predict(input_img, new_model)

# coding=utf-8
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
import time
from tqdm import tqdm
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras import backend as K
K.set_image_dim_ordering('tf')


from semantic_segmentation_networks import binary_unet, binary_fcnnet, binary_segnet, binary_unet_onlyjaccard, binary_unet_jaccard
from ulitities.base_functions import load_img_normalization,  load_img_by_gdal, UINT16, UINT8, UINT10

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 7
np.random.seed(seed)



img_w = 256
img_h = 256

n_label = 1

im_bands =4
im_type = UINT10  # UINT8:0, UINT10:1, UINT16:2

dict_network = {0: 'unet', 1: 'fcnnet', 2: 'segnet'}
dict_target = {0: 'roads', 1: 'buildings'}

FLAG_USING_NETWORK = 0  # 0:unet; 1:fcn; 2:segnet;
FLAG_TARGET_CLASS = 1   # 0:roads; 1:buildings
FLAG_MAKE_TEST=True

date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
print("date and time: {}".format(date_time))

base_model = ""

# model_save_path = ''.join(['../../data/models/sat_urban_4bands/',dict_network[FLAG_USING_NETWORK], '_',
#                            dict_target[FLAG_TARGET_CLASS], '_binary_notonehot_', date_time, '.h5'])
model_save_path = ''.join(['/home/omnisky/PycharmProjects/data/models/ssj/shuidao_jaccard', date_time, '.h5'])
print("model save as to: {}".format(model_save_path))

# train_data_path = ''.join(['../../data/traindata/sat_urban_4bands/binary/',dict_target[FLAG_TARGET_CLASS], '/'])
#
train_data_path = '/home/omnisky/PycharmProjects/data/traindata/shuidao/'
print("traindata from: {}".format(train_data_path))

"""get the train file name and divide to train and val parts"""
def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(train_data_path + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            # img = load_img(train_data_path + 'src/' + url)

            _, img = load_img_normalization(im_bands, (train_data_path + 'src/' + url), data_type=im_type)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            train_data.append(img)
            # label = load_img(train_data_path + 'label/' + url, grayscale=True)
            _, label = load_img_normalization(1, (train_data_path + 'label/' + url))
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                # train_label = to_categorical(train_label, num_classes=n_label)  # one_hot coding
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            # img = load_img(train_data_path + 'src/' + url)
            _, img = load_img_normalization(im_bands, (train_data_path + 'src/' + url), data_type=im_type)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            valid_data.append(img)
            # label = load_img(train_data_path + 'label/' + url, grayscale=True)
            _, label = load_img_normalization(1, (train_data_path + 'label/' + url))
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                # valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0



""" For tes multiGPU model"""
import keras
from keras.utils import multi_gpu_model
class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self,  path):
        # self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))

        self.model.save_weights(self.path, overwrite=True)

        self.best_loss = val_loss

"""Train model ............................................."""
def train(model):
    EPOCHS = 100  # should be 10 or bigger number
    BS = 32

    if os.path.isfile(base_model):
        model.load_weights(base_model)
        print("load last weight from:{}".format(base_model))

    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc', save_best_only=True, mode='max')
    model_earlystop=EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')

    # model_checkpoint = ModelCheckpoint(
    #     model_save_path,
    #     monitor='val_jaccard_coef_int',
    #     save_best_only=False)

    # model_checkpoint = ModelCheckpoint(
    #     model_save_path,
    #     monitor='val_jaccard_coef_int',
    #     save_best_only=True,
    #     mode='max')

    # model_earlystop = EarlyStopping(
    #     monitor='val_jaccard_coef_int',
    #     patience=6,
    #     verbose=0,
    #     mode='max')

    """自动调整学习率"""
    model_reduceLR=ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.1,
        patience=3,
        verbose=0,
        mode='max',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )

    model_history = History()

    callable = [model_checkpoint,model_earlystop, model_reduceLR, model_history]

    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print ("the number of train data is", train_numb)
    print ("the number of val data is", valid_numb)


    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=callable, max_q_size=1)

    #plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.plot(np.arange(0, N), H.history["jaccard_coef_int"], label="train_jaccard_coef_int")
    plt.plot(np.arange(0, N), H.history["val_jaccard_coef_int"], label="val_jaccard_coef_int")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    fig_train_acc = ''.join(['../../data/models/train_acc_', dict_network[FLAG_USING_NETWORK], '_',
                           dict_target[FLAG_TARGET_CLASS], '.png'])
    plt.savefig(fig_train_acc)



"""
Test the model which has been trained right now
"""
window_size=256

def test_predict(bands, image,model):
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
            print('crop:{}'.format(crop.shape))

            pred = model.predict(crop, verbose=2)
            # pred = np.argmax(pred, axis=2)  #for one hot encoding
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1

            pred = pred.reshape(256, 256)
            print(np.unique(pred))

            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    outputresult =mask_whole[0:h,0:w]
    # outputresult = outputresult.astype(np.uint8)

    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    cv2.imwrite('../../data/predict/test_model_notonehot.png',outputresult*255)
    return outputresult




if __name__ == '__main__':

    if not os.path.isdir(train_data_path):
        print ("train data does not exist in the path:\n {}".format(train_data_path))

    if FLAG_USING_NETWORK==0:
        # model = binary_unet(im_bands,n_label)
        model = binary_unet_jaccard(img_w, img_h, im_bands, n_label)
    elif FLAG_USING_NETWORK==1:
        model = binary_fcnnet(img_w, img_h, im_bands, n_label)
    elif FLAG_USING_NETWORK==2:
        model=binary_segnet(img_w, img_h, im_bands, n_label)

    print("Train by : {}".format(dict_network[FLAG_USING_NETWORK]))
    train(model)

    if FLAG_MAKE_TEST:
        print("test ....................predict by trained model .....\n")
        test_img_path = '../../data/test/sample1.png'
        import sys

        if not os.path.isfile(test_img_path):
            print("no file: {}".format(test_img_path))
            sys.exit(-1)

        input_img = load_img_by_gdal(test_img_path)
        if im_type == UINT8:
            input_img = input_img / 255.0
        elif im_type == UINT10:
            input_img = input_img / 1024.0
        elif im_type == UINT16:
            input_img = input_img / 65535.0
        input_img = np.clip(input_img, 0.0, 1.0)

        new_model = load_model(model_save_path)

        test_predict(input_img, new_model)

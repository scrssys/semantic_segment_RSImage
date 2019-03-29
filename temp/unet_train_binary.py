# coding=utf-8
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras import backend as K
# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256

n_label = 1+1



"""for roads"""
# model_save_path = '../../data/models/unet_roads.h5'
# train_data_path = '../../data/traindata/binary/roads/'

"""for buildings"""
model_save_path = '../../data/models/unet_buildings_onehot.h5'
train_data_path = '../../data/traindata/all/binary/buildings/'




def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0  # MY image preprocessing
    return img


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
            img = load_img(train_data_path + 'src/' + url)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(train_data_path + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                train_label = to_categorical(train_label, num_classes=n_label)
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
            img = load_img(train_data_path + 'src/' + url)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(train_data_path + 'label/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def unet():
    # inputs = Input((3, img_w, img_h))  # channels_first
    inputs = Input((img_w, img_h, 3))

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    # up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=1)
    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    # up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    # up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    # up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    conv10 = Reshape((img_w * img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



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



def train():
    EPOCHS = 10  # should be 10 or bigger number
    BS = 16

    model = unet()

    """test the model fastly but can only train one epoch, AND it does not work finally ^_^"""
    # model = multi_gpu_model(model, gpus=4)
    # model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    ##### modelcheck = [CustomModelCheckpoint('./data/models/unet_fff.h5')]

    modelcheck = ModelCheckpoint(model_save_path, monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
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
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('../../data/models/unet_binary_train.png')


"""
only test
"""
window_size=256

def unet_predict(image,model):
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

            # pred = model.predict_classes(crop, verbose=2)
            # pred = model.predict(crop, verbose=2)
            pred = model.predict(crop, verbose=2)
            pred = np.argmax(pred,axis=2)  #for one hot encoding

            pred = pred.reshape(256, 256)
            print(np.unique(pred))


            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    # outputresult = mask_whole[0:h, 0:w] * 255.0
    outputresult =mask_whole[0:h,0:w]
    outputresult = outputresult.astype(np.uint8)


    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    cv2.imwrite('../../data/predict/test_onehot.png',outputresult*255)
    return outputresult

if __name__ == '__main__':

    if not os.path.isdir(train_data_path):
        print ("train data does not exist in the path:\n {}".format(train_data_path))

    train()

    # print("test ....................predict by unet one hot encoding .....\n")
    # img_path = '../../data/test/1.png'
    # import sys
    #
    # if not os.path.isfile(img_path):
    #     print("no file: {}".forma(img_path))
    #     sys.exit(-1)
    #
    # input_img = cv2.imread(img_path)
    # input_img = np.array(input_img, dtype="float") / 255.0  # must do it
    #
    # new_model = load_model(model_save_path)
    #
    # unet_predict(input_img, new_model)
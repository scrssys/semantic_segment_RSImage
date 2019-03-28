#coding:utf-8


from keras import applications

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape,\
    Permute, Activation, Input, Conv2DTranspose,Dropout,concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras import backend as K
K.set_image_dim_ordering('tf')


img_w = 256
img_h = 256

n_label = 2

window_size=256  # for predict test


"""for buildings"""
model_save_path = '../../data/models/fcn8net_buildings.h5'
train_data_path = '../../data/traindata/all/binary/buildings/'



def FCN8net():
    # inputs = Input((3, img_w, img_h))  # channels_first
    inputs = Input((img_w, img_h, 3))

    # Block 1

    block1_conv1 = Conv2D(64,(3, 3),activation='relu',padding='same')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1_conv1)
    block1_pool=MaxPooling2D((2, 2), strides=(2, 2))(block1_conv2)



    # Block 2
    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2))(block2_conv2)


    # Block 3
    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2))(block3_conv3)



    # Block 4
    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2))(block4_conv3)



    # Block 5
    block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2))(block5_conv3)


    # The following blocks were added by qiaozh--------------------------------------------

    # Block6
    block6_conv = Conv2D(4096,(7,7), activation='relu', padding='same')(block5_pool)
    block6_drop = Dropout(0.5)(block6_conv)

    # Block7
    block7_conv = Conv2D(4096, (1, 1), activation='relu', padding='same')(block6_drop)
    block7_drop = Dropout(0.5)(block7_conv)

    # fcn8
    fcn8 = Conv2D(n_label, (1, 1), activation='relu', padding='same')(block7_drop)

    # up9
    fcn9=Conv2DTranspose(512,(4,4),strides=(2,2), padding='same')(fcn8)
    up9 =concatenate([fcn9,block4_pool],axis=3)

    #up10
    fcn10 = Conv2DTranspose(256,(4,4), strides=(2,2), padding='same')(up9)
    up10 = concatenate([fcn10,block3_pool],axis=3)

    #up11
    up11 = Conv2DTranspose(n_label,(16,16), strides=(8,8), padding='same')(up10)
    up11 = Reshape((img_w * img_h, n_label))(up11)  # 4D(bath_size, img_w*img_h, n_label)

    model = Model(inputs=inputs, outputs=up11)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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



def train():
    EPOCHS = 10  # should be 10 or bigger number
    BS = 16

    model = FCN8net()



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



def fcn8_predict(image,model):
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
            # crop = padding_img[:3, i * stride:i * stride + window_size, j * stride:j * stride + window_size]
            crop = padding_img[i * stride:i * stride + window_size, j * stride:j * stride + window_size, :3]

            crop = np.expand_dims(crop, axis=0)
            print('crop:{}'.format(crop.shape))

            pred = model.predict(crop, verbose=2)
            # pred = np.argmax(pred,axis=2)  #for one hot encoding

            pred = pred.reshape(256, 256)
            print(np.unique(pred))


            mask_whole[i * stride:i * stride + window_size, j * stride:j * stride + window_size] = pred[:, :]

    # outputresult = mask_whole[0:h, 0:w] * 255.0
    outputresult =mask_whole[0:h,0:w] + 0.5
    outputresult = outputresult.astype(np.uint8)


    plt.imshow(outputresult, cmap='gray')
    plt.title("Original predicted result")
    plt.show()
    cv2.imwrite('../../data//predict/fcn/not_onehot.png',outputresult*255)
    return outputresult


if __name__=="__main__":
    print("Training by FCN-8...\n")

    # train()

    print("test ....................predict by FCN-8 .....\n")
    img_path = '../../data/test/H48G036036AP_Clip00.png'
    import sys

    if not os.path.isfile(img_path):
        print("no file: {}".forma(img_path))
        sys.exit(-1)

    input_img = cv2.imread(img_path)
    input_img = np.array(input_img, dtype="float") / 255.0  # must do it

    new_model = load_model(model_save_path)

    fcn8_predict(input_img, new_model)





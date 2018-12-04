# coding:utf-8

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras import metrics, losses
from keras.optimizers import Nadam
from keras import optimizers

from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, History,ReduceLROnPlateau

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

from keras.utils.np_utils import to_categorical
from semantic_segmentation_networks import binary_unet_jaccard, binary_fcnnet_jaccard, binary_segnet_jaccard
from ulitities.base_functions import load_img_normalization, load_img_by_gdal, UINT16, UINT8, UINT10


sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

"""get the train file name and divide to train and val parts"""
def get_train_val(train_data_path, val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(train_data_path + '/src'):
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
def generateData(batch_size,im_bands, im_type, train_data_path, img_w, img_h, n_label, target_value, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            _, img = load_img_normalization(im_bands, (train_data_path + '/src/' + url), data_type=im_type)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            train_data.append(img)
            _, label = load_img_normalization(1, (train_data_path + '/label/' + url))
            label = img_to_array(label)
            indx = np.where(label==target_value)
            label[:,:] = 0
            label[indx] = 1

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
def generateValidData(batch_size,im_bands,im_type,  train_data_path, img_w, img_h, n_label, target_value,data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            _, img = load_img_normalization(im_bands, (train_data_path + '/src/' + url), data_type=im_type)

            # Adapt dim_ordering automatically
            img = img_to_array(img)
            valid_data.append(img)
            _, label = load_img_normalization(1, (train_data_path + '/label/' + url))
            label = img_to_array(label)
            indx = np.where(label == target_value)
            label[:, :] = 0
            label[indx] = 1

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


def train_binary_for_ui(input_dict={}):
    # self.input_dict = input_dict

    if os.path.isdir(input_dict['trainData_path']):
        train_data_path = input_dict['trainData_path']
    else:
        sys.exit(-2)

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print("date and time: {}".format(date_time))

    if os.path.isdir(input_dict['saveModel_path']):
        model_path = ''.join([input_dict['saveModel_path'], '/', input_dict['network'], '_',
                              input_dict['class_name'], '_binary_', input_dict['target_function'], '_',
                              str(input_dict['windsize']), '_', date_time, '.h5'])
        saveModel_path = input_dict['saveModel_path']
    else:
        sys.exit(-3)
    print("save model to:{}".format(model_path))

    base_model = input_dict['baseModel']
    network = input_dict['network']
    if 'unet' in base_model:
        assert ('unet' in network)
    elif 'fcnnet' in base_model:
        assert ('fcnnet' in network)
    elif 'segnet' in base_model:
        assert ('segnet' in network)

    target_function = input_dict['target_function']

    label_value = input_dict['label_value']


    if input_dict['im_bands'] > 0:
        im_bands = input_dict['im_bands']
    else:
        sys.exit(-4)
    im_type = input_dict['dtype']

    if input_dict['windsize'] > 32:
        img_w = input_dict['windsize']
        img_h = input_dict['windsize']
    if input_dict['BS'] > 1:
        BS = input_dict['BS']
    if input_dict['EPOCHS'] > 1:
        EPOCHS = input_dict['EPOCHS']
    # target_class = input_dict['target_class']
    # if target_class not in train_data_path:
    #     print("target class and train data path is not consistent!")
    #     sys.exit(-5)

    gup_id = input_dict['GPUID']
    os.environ["CUDA_VISIBLE_DEVICES"] = gup_id

    n_label = 1

    if 'unet' in network:
        if 'crossentropy' in target_function:
            model = binary_unet(img_w, img_h, im_bands, n_label)
        elif 'jaccard' in target_function:
            model = binary_unet_onlyjaccard(img_w, img_h, im_bands, n_label)
        else:
            model = binary_unet_jaccard(img_w, img_h, im_bands, n_label)
    elif 'fcnnet' in network:
        if 'crossentropy' in target_function:
            model = binary_fcnnet(img_w, img_h, im_bands, n_label)
        elif 'jaccard' in target_function:
            model = binary_fcnnet_onlyjaccard(img_w, img_h, im_bands, n_label)
        else:
            model = binary_fcnnet_jaccard(img_w, img_h, im_bands, n_label)
    elif 'segnet' in network:
        if 'crossentropy' in target_function:
            model = binary_segnet(img_w, img_h, im_bands, n_label)
        elif 'jaccard' in target_function:
            model = binary_segnet_onlyjaccard(img_w, img_h, im_bands, n_label)
        else:
            model = binary_segnet_jaccard(img_w, img_h, im_bands, n_label)

    if os.path.isfile(base_model):
        print("load last weight from:{}".format(base_model))
        model.load_weights(base_model)

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_jaccard_coef_int',
        save_best_only=False)

    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef_int',
        patience=6,
        verbose=0,
        mode='max')

    """自动调整学习率"""
    model_reduceLR = ReduceLROnPlateau(
        monitor='val_jaccard_coef_int',
        factor=0.1,
        patience=3,
        verbose=0,
        mode='max',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )

    model_history = History()

    callable = [model_checkpoint, model_earlystop, model_reduceLR, model_history]

    train_set, val_set = get_train_val(train_data_path)
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)

    H = model.fit_generator(generator=generateData(BS,im_bands,im_type, train_data_path, img_w, img_h, n_label, label_value,train_set),
                            steps_per_epoch=train_numb // BS,
                            epochs=EPOCHS, verbose=1,
                            validation_data=generateValidData(BS,im_bands,im_type, train_data_path, img_w, img_h, n_label, label_value, val_set),
                            validation_steps=valid_numb // BS,
                            callbacks=callable)

    # # plot the training loss and accuracy
    # plt.style.use("ggplot")
    # plt.figure()
    # N = EPOCHS
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    # plt.plot(np.arange(0, N), H.history["jaccard_coef_int"], label="train_jaccard_coef_int")
    # plt.plot(np.arange(0, N), H.history["val_jaccard_coef_int"], label="val_jaccard_coef_int")
    # plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # fig_train_acc = ''.join([saveModel_path, network, '_',
    #                          input_dict['class_name'], '_jaccard.png'])
    # plt.savefig(fig_train_acc)

    return 0


def binary_unet(img_w, img_h, im_bands, n_label=1):
    inputs = Input((img_w, img_h, im_bands))

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
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    conv10 = Reshape((img_w * img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


    model.summary()
    return model

def binary_unet_jaccard(img_w, img_h, im_bands, n_label=1):
    inputs = Input((img_w, img_h, im_bands))

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
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    conv10 = Reshape((img_w * img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)
    # adam = Adam()

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])

    # model.compile(optimizer='Adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    model.summary()
    return model


def binary_unet_onlyjaccard(img_w, img_h, im_bands, n_label=1):
    inputs = Input((img_w, img_h, im_bands))

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
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    conv10 = Reshape((img_w * img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)
    # adam = Adam()

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])

    # model.compile(optimizer='Adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    model.summary()
    return model

def multiclass_unet(img_w, img_h, im_bands, n_label=3):
    inputs = Input((img_w, img_h, im_bands))

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
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    conv10 = Reshape((img_w*img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def multiclass_unet_jaccard(img_w, img_h,im_bands, n_label=3):
    inputs = Input((img_w, img_h, im_bands))

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
    drop4 = Dropout(0.5)(conv4) # add 20180621
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    conv10 = Reshape((img_w*img_h, n_label))(conv10)  # 4D(bath_size, img_w*img_h, n_label)

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd,  # Nadam(lr=1e-3)
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    model.summary()
    return model

def binary_fcnnet(img_w, img_h, im_bands, n_label=2):

    inputs = Input((img_w, img_h, im_bands))

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

def binary_fcnnet_jaccard(img_w, img_h, im_bands, n_label=1):

    inputs = Input((img_w, img_h, im_bands))

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
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    return model

def binary_fcnnet_onlyjaccard(img_w, img_h, im_bands, n_label=1):

    inputs = Input((img_w, img_h, im_bands))

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
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])
    return model


def multiclass_fcnnet(img_w, img_h, im_bands,n_label=3):

    inputs = Input((img_w, img_h, im_bands))

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
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def multiclass_fcnnet_jaccard(img_w, img_h, im_bands,n_label=3):

    inputs = Input((img_w, img_h, im_bands))

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
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    return model

def binary_segnet(img_w, img_h, im_bands, n_label=2):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))

    model.add(Activation('sigmoid')) # for test one class

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

def binary_segnet_jaccard(img_w, img_h, im_bands,n_label=1):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))

    model.add(Activation('sigmoid')) # for test one class

    # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])
    model.summary()
    return model


def binary_segnet_onlyjaccard(img_w, img_h, im_bands,n_label=1):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))

    model.add(Activation('sigmoid')) # for test one class

    # model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])
    model.summary()
    return model


def multiclass_segnet(img_w, img_h, im_bands, n_label=3):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu')) # for channels_last
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu')) # for channels_last
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))

    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2) # for theano backend
    # model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
    return model

def multiclass_segnet_jaccard(img_w, img_h, im_bands,n_label=3):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu')) # for channels_last
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, im_bands), padding='same', activation='relu')) # for channels_last
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Reshape((img_w * img_h, n_label)))

    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2) # for theano backend
    # model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    model.summary()
    return model

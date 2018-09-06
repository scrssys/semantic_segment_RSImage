# coding:utf-8
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, \
    Reshape, BatchNormalization,Permute, Activation, Input,Conv2DTranspose

from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.optimizers import *
from keras.layers.merge import concatenate


from keras import metrics, losses
from keras.optimizers import Nadam
from keras import optimizers

img_w=256
img_h=256

# K.set_value(sgd.lr, 0.2 * K.get_value(sgd.lr))

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# adam = optimizers.Adam(lr=0.01, decay=1e-6)
# nadam = optimizers.Nadam

def binary_unet(im_bands, n_label=1):
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
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


    model.summary()
    return model


def binary_unet_4orMore(im_bands, n_label=2):
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
    # model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def binary_unet_jaccard(im_bands, n_label=1):
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


def binary_unet_jaccard_4orMore(im_bands, n_label=1):
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

    model.compile(optimizer=sgd,
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef_int])

    model.summary()
    return model


def binary_unet_jaccard_notOnehot(n_label=1):
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

    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,   #Nadam(lr=1e-3)
                  loss=losses.jaccard_coef_binary_crossentropy_loss,
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])

    # model.compile(optimizer='Adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    model.summary()
    return model


def multiclass_unet(im_bands, n_label=3):
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
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def multiclass_unet_jaccard(n_label=3):
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

def binary_fcnnet(im_bands, n_label=2):

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

def binary_fcnnet_jaccard(im_bands, n_label=1):

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

def multiclass_fcnnet(im_bands,n_label=3):

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

def multiclass_fcnnet_jaccard(n_label=3):

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
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.jaccard_coef, metrics.jaccard_coef_int])
    return model


def binary_segnet(im_bands, n_label=2):
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

def binary_segnet_jaccard(im_bands,n_label=1):
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


def multiclass_segnet(im_bands, n_label=3):
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

def multiclass_segnet_jaccard(n_label=3):
    model = Sequential()
    #encoder
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='relu')) # for channels_last
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
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h, 3), padding='same', activation='relu')) # for channels_last
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
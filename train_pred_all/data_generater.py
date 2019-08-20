
import numpy as np
import random
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from scipy.signal import medfilt, medfilt2d
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import sys

from ulitities.base_functions import load_img_bybandlist, load_img_normalization,UINT8,UINT10,UINT16


def random_crop(img1, img2, crop_H, crop_W):

    # assert  img1.size[:2] ==  img2.size[:2]
    # print(img1.shape[:2])
    assert(img1.shape[:2] == img2.shape[:2])
    h, w = img1.shape[:2]

    # 裁剪宽度不可超过原图可裁剪宽度
    if crop_W > w:
        crop_W = w
        print("crop width is lager than img width")
    # 裁剪高度

    if crop_H > h:
        crop_H = h
        print("crop height is lager than img height")

    # 随机生成左上角的位置
    x0 = random.randrange(0, w - crop_W + 1, 50)
    y0 = random.randrange(0, h - crop_H + 1, 50)

    crop_1 = img1[y0:y0+crop_H,x0:x0+crop_W, :]
    crop_2 = img2[y0:y0+crop_H,x0:x0+crop_W]

    return crop_1,crop_2

def rotate(xb, yb, angle):
    xb = np.rot90(np.array(xb), k=angle)

    yb = np.rot90(np.array(yb), k=angle)

    return xb, yb

def add_noise( xb, width, height, dtype=1):
    assert(xb.shape[-1]<xb.shape[0])
    if dtype == 1:
        noise_value = 255
    elif dtype == 2:
        noise_value = 1024
    else:
        noise_value = 65535

    tmp = np.random.random() / 20.0  # max = 0.05
    noise_num = int(tmp * width * height)
    for i in range(noise_num):
        temp_x = np.random.randint(0, xb.shape[0])
        temp_y = np.random.randint(0, xb.shape[1])
        xb[temp_x, temp_y,:] = noise_value
        # xb[:, temp_x, temp_y] = noise_value
    return xb


def gamma_tansform(xb, g=2.0):
    tmp = np.random.random() * g
    # print("gamma:{}".format(tmp))
    if tmp < 0.6:
        tmp = 0.6
    if tmp > 1.4:
        tmp = 1.4

    xb = exposure.adjust_gamma(xb, tmp)
    return xb

def med_filtering(xb, w=3):
    xb = xb.astype(np.float32)
    # a, b, c = xb.shape
    # if a < c:
    #     xb = xb.transpose(1, 2, 0)
    _, _, bands = xb.shape

    for i in range(bands):
        xb[:, :, i] = medfilt2d(xb[:, :, i], (w, w))
    # if a < c:
    #     xb = np.transpose(xb, (2, 0, 1))
    xb = xb.astype(np.uint16)
    return xb

def data_augment(xb, yb, w, h, d_type=1):
    if np.random.random() < 0.25:
        assert (yb.shape[0] == yb.shape[1])
        assert (xb.shape[0] == xb.shape[1])
        xb, yb = rotate(xb, yb, 1)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 2)

    if np.random.random() < 0.25:
        assert (yb.shape[0] == yb.shape[1])
        assert (xb.shape[0] == xb.shape[1])
        xb, yb = rotate(xb, yb, 3)

    if np.random.random() < 0.25:
        # xb = np.transpose(xb, (1, 2, 0))
        xb = np.fliplr(xb)  # flip an array horizontally
        # xb = np.transpose(xb, (2, 0, 1))
        yb = np.fliplr(yb)

    if np.random.random() < 0.25:
        # xb = np.transpose(xb, (1, 2, 0))
        xb = np.flipud(xb)  # flip an array vertically (up down directory)
        # xb = np.transpose(xb, (2, 0, 1))
        yb = np.flipud(yb)

    if np.random.random() < 0.25:  # gamma adjust
        tmp = np.random.random() * 2
        xb = gamma_tansform(xb,tmp)

    if np.random.random() < 0.25:  # medium filtering
        xb = med_filtering(xb,3)

    if np.random.random() < 0.2:
        xb = add_noise(xb, w, h, d_type)
    return xb, yb


def resample_data(img, dst_h, dst_w, mode = Image.ANTIALIAS, bits=8):
    if len(img.shape)>2:
        if bits==8:
            n_img = np.zeros((dst_h, dst_w, img.shape[-1]), np.uint8)
            img = np.asarray(img, np.uint8)
            for i in range(img.shape[-1]):
                b_img = img[:, :, i]
                b_img = Image.fromarray(b_img, mode='L')

                b_img = b_img.resize((dst_h, dst_w), mode)
                b_img = np.array(b_img, np.uint8)
                n_img[:, :, i] = b_img[:, :]
            return n_img
        else:
            n_img = np.zeros((dst_h, dst_w, img.shape[-1]), np.uint16)
            img = np.asarray(img, np.uint32)
            for i in range(img.shape[-1]):
                b_img = img[:, :, i]
                # plt.figure()
                # plt.imshow(b_img, cmap='gray')
                # plt.show()
                b_img = Image.fromarray(b_img, mode='I')
                b_img = b_img.resize((dst_h, dst_w), mode)
                b_img = np.array(b_img, np.uint16)
                n_img[:, :, i] = b_img[:, :]
            return n_img
    else:
        img = Image.fromarray(img, mode='L')
        img = img.resize((dst_h, dst_w), mode)
        img = np.array(img, np.uint8)
        return img


# data for training
def train_data_generator(config, sample_url):
    # print 'generateData...'
    norm_value =255.0
    bits_num=8
    if '10' in config.im_type:
        norm_value = 1024.0
        bits_num = 16
    elif '16' in config.im_type:
        norm_value = 65535.0
        bits_num = 16
    else:
        pass
    label_list,img_list=[],[]
    for pic in sample_url:
        _,t_img = load_img_normalization(1,config.train_data_path+'/label/'+pic)
        tp = np.unique(t_img)
        if len(tp) < 2:
            print("Only one value {} in {}".format(tp, config.train_data_path+'/label/'+pic))
            if tp[0] == 0:
                print("no target value in {}".format(config.train_data_path+'/label/'+pic))
                continue

        label_list.append(t_img)
        ret, s_img = load_img_bybandlist((config.train_data_path + '/src/' + pic),bandlist=config.band_list)
        if ret!=0:
            continue

        s_img=img_to_array(s_img)
        s_img = np.asarray(s_img, np.uint16)
        # plt.imshow(s_img[:,:,0])
        # plt.show()
        img_list.append(s_img)
    assert len(label_list) == len(img_list)

    while True:
        train_data = []
        train_label = []
        batch = 0

        for i in np.random.permutation(np.arange(len(img_list))):
            src_img=img_list[i]
            label_img=label_list[i]
            random_size = random.randrange(config.img_w, config.img_w*2+1, config.img_w)
            # random_size = 960
            img, label = random_crop(src_img, label_img, random_size, random_size)

            if config.label_nodata in np.unique(label):
                continue
            """ignore pure background area"""
            if len(np.unique(label)) < 2:
                if (0 in np.unique(label)) and (np.random.random() < 0.75):
                    continue

            if img.shape[1] != config.img_w or img.shape[0] != config.img_h:
                # print("resize samples")
                img = resample_data(img,config.img_h,config.img_w,mode=Image.BILINEAR, bits=bits_num)
                label=resample_data(label, config.img_h, config.img_w,mode=Image.NEAREST)

            if config.augment:
                img, label = data_augment(img,label,config.img_w,config.img_h)

            img = np.asarray(img).astype(np.float32)/norm_value
            img = np.clip(img, 0.0, 1.0)

            batch +=1
            img = img_to_array(img)
            label=img_to_array(label)
            train_data.append(img)
            train_label.append(label)
            if batch%config.batch_size==0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                print("img shape:{}".format(train_data.shape))
                print("label shap:{}".format(train_label.shape))
                if config.nb_classes > 2:
                    train_label = to_categorical(train_label, num_classes=config.nb_classes)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


def val_data_generator(config, sample_url):
    # print 'generate validating Data...'
    norm_value = 255.0
    w=config.img_w
    h=config.img_h
    label_list, img_list = [],[]
    for pic in sample_url:
        _, t_img = load_img_normalization(1, config.train_data_path + '/label/' + pic)
        label_list.append(t_img)
        tp = np.unique(t_img)
        if len(tp) < 2:
            print("Only one value {} in {}".format(tp, config.train_data_path + '/label/' + pic))
            if tp[0] == 0:
                print("no target value in {}".format(config.train_data_path + '/label/' + pic))
                continue
        ret, s_img = load_img_bybandlist((config.train_data_path + '/src/' + pic), bandlist=config.band_list)
        if ret!=0:
            continue
        s_img = img_to_array(s_img)
        img_list.append(s_img)

    assert len(label_list) == len(img_list)

    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(img_list))):
            img = img_list[i]
            label = label_list[i]
            img = np.asarray(img).astype("float") / norm_value
            label = np.asarray(label)
            assert img.shape[0:2] == label.shape[0:2]

            for i in range(img.shape[0] // h):
                for j in range(img.shape[1] // w):
                    x = img[i * h:(i + 1) * h, (j * w):(j + 1) * w, :]
                    y = label[i * h:(i + 1) * h, (j * w):(j + 1) * w]

                    if config.label_nodata in np.unique(y):
                        continue
                    """ignore pure background area"""
                    if len(np.unique(y)) < 2:
                        if (0 in np.unique(y)) and (np.random.random() < 0.75):
                            continue
                    x = img_to_array(x)
                    y = img_to_array(y)
                    train_data.append(x)
                    train_label.append(y)

                    batch += 1
                    if batch % config.batch_size == 0:
                        train_data = np.array(train_data)
                        train_label = np.array(train_label)
                        if config.nb_classes > 2:
                            train_label = to_categorical(train_label, num_classes=config.nb_classes)
                        yield (train_data, train_label)
                        train_data = []
                        train_label = []
                        batch = 0
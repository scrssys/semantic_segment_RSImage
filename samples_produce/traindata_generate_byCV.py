# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import sys


from ulitities.base_functions import get_file

# seed = 1
# np.random.seed(seed)

img_w = 256
img_h = 256

valid_labels=[0,1]

# FLAG_BINARY = False
FLAG_BINARY = True


input_path = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/originalLabelandImages/rice/'


output_path = '../../data/traindata/rice/'

if os.path.isdir(output_path):
    print("ok")
    pass


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


""" check the size of src_img and label_img"""
def check_src_label_size(srcimg, labelimg):
    row_src, column_src,_ = srcimg.shape
    print("source image size: ({},{})".format(row_src, column_src))
    row_label, column_label = labelimg.shape
    print("label image size: ({},{})".format(row_label, column_label))
    assert (row_src==row_label and column_src==column_src)


"""check some invalid labels or NoData values"""
def check_invalid_labels(img, labels):
    local_labels = np.unique(img)
    for ll in local_labels:
        assert(ll in labels)


def creat_dataset_multiclass(in_path, out_path, image_num=50000, mode='original'):

    print('\ncreating dataset...')

    label_files,tt = get_file(os.path.join(in_path,'label/'))
    assert(tt!=0)

    image_each = image_num/len(label_files)

    g_count = 0
    for label_file in tqdm(label_files):
        count = 0
        src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
        if not os.path.isfile(src_file):
            print("Have no file:".format(src_file))
            continue
            # sys.exit(-1)

        print("src file:{}".format(os.path.split(src_file)[1]))
        src_img = cv2.imread(src_file)

        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        """Check image size and invalid labels"""
        check_src_label_size(src_img, label_img)
        # check_invalid_labels(label_img, valid_labels)

        X_height, X_width, _ = src_img.shape

        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

            """ignore nodata area"""
            FLAG_HAS_NODATA=False
            tmp= np.unique(label_roi)
            for tt in tmp:
                if tt not in valid_labels:
                    FLAG_HAS_NODATA =True
                    continue

            if FLAG_HAS_NODATA==True:
                continue


            # """Cut down the pure background image with 80% probability"""
            # if len(np.unique(label_roi)) < 2:
            #     if np.unique(label_roi)[0] ==0:
            #         if np.random.random()< 0.8:
            #             continue

            """ignore whole background area"""
            if len(np.unique(label_roi)) < 2:
                if 0 in np.unique(label_roi):
                    continue

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/src/%d.png' % g_count), src_roi)
            cv2.imwrite((out_path + '/label/%d.png' % g_count), label_roi)
            count += 1
            g_count += 1


def creat_dataset_binary(in_path, out_path, image_num=50000, mode='original'):
    print('\ncreating dataset...')

    label_files,tt = get_file(os.path.join(in_path,'label/'))
    assert(tt!=0)

    image_each = image_num/len(label_files)

    g_count = 0
    for label_file in tqdm(label_files):

        src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
        if not os.path.isfile(src_file):
            print("Have no file:".format(src_file))
            continue
            # sys.exit(-1)

        print("src file:{}".format(os.path.split(src_file)[1]))
        src_img = cv2.imread(src_file)

        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        """Check image size and invalid labels"""
        check_src_label_size(src_img, label_img)
        # check_invalid_labels(label_img, valid_labels)

        X_height, X_width, _ = src_img.shape


        print("\n1: produce road labels---------------------")
        index = np.where(label_img == 1)  # 1: roads
        road_label = np.zeros((X_height, X_width), np.uint8)
        road_label[index] = 1

        print(np.unique(road_label))
        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = road_label[random_height: random_height + img_h, random_width: random_width + img_w]

            """ignore nodata area"""
            FLAG_HAS_NODATA = False
            tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
            for tt in tmp:
                if tt not in valid_labels:
                    FLAG_HAS_NODATA = True
                    continue

            if FLAG_HAS_NODATA==True:
                continue

            # """Cut down the pure background image with 80% probability"""
            # if len(np.unique(label_roi)) < 2:
            #     if np.unique(label_roi)[0] ==0:
            #         if np.random.random()< 0.8:
            #             continue

            """ignore pure background area"""
            if len(np.unique(label_roi)) < 2:
                if 0 in np.unique(label_roi):
                    continue

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/roads/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/roads/src/%d.png' % g_count), src_roi)
            cv2.imwrite((out_path + '/roads/label/%d.png' % g_count), label_roi)
            count += 1
            g_count += 1


        print("\n2: produce buildings labels---------------------")
        index = np.where(label_img == 2)  # 1: buildings
        building_label = np.zeros((X_height, X_width), np.uint8)
        building_label[index] = 1

        count = 0
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = building_label[random_height: random_height + img_h, random_width: random_width + img_w]

            """ignore nodata area"""
            FLAG_HAS_NODATA = False
            tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
            for tt in tmp:
                if tt not in valid_labels:
                    FLAG_HAS_NODATA = True
                    continue

            if FLAG_HAS_NODATA == True:
                continue

            """Cut down the pure background image with 80% probability"""
            # if len(np.unique(label_roi)) < 2:
            #     if np.unique(label_roi)[0] == 0:
            #         if np.random.random() < 0.8:
            #             continue

            """ignore pure background area"""
            if len(np.unique(label_roi)) < 2:
                if 0 in np.unique(label_roi):
                    continue

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            visualize = label_roi * 50

            cv2.imwrite((out_path + '/buildings/visualize/%d.png' % g_count), visualize)
            cv2.imwrite((out_path + '/buildings/src/%d.png' % g_count), src_roi)
            cv2.imwrite((out_path + '/buildings/label/%d.png' % g_count), label_roi)
            count += 1
            g_count += 1

if __name__ == '__main__':

    """check input directories"""
    if not os.path.isdir(os.path.join(input_path,'src/')):
        print("No input src directory:{}".format(os.path.join(input_path,'src/')))
        sys.exit(-1)
    if not os.path.isdir(os.path.join(input_path,'label/')):
        print("No input label directory:{}".format(os.path.join(input_path,'label/')))
        sys.exit(-2)


    if FLAG_BINARY == True:
        output_path = ''.join([output_path, '/binary/'])
    else:
        output_path = ''.join([output_path, '/multiclass/'])


    if FLAG_BINARY==True:
        print("Produce labels for binary classification")
        creat_dataset_binary(input_path, output_path, 300000, mode='augment')
    else:
        print("produce labels for multiclass")
        creat_dataset_multiclass(input_path, output_path, 100000, mode='augment')
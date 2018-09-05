#coding:utf-8

import numpy as np
import cv2
import os
import gc
import sys
import random
import tensorflow as tf

from ulitities.base_functions import load_img

# seed = 1
# random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

valid_labels=[0,1,2]

ref_file = '../../data/tmp/sample1.png'  # sample1, shuangliu_1test_label, yushui22_test_label
pred_file = '../../data/predict/unet/sat_nrg/unet_binary_jaccard_combined_sample1_nrg.png'
check_rate=0.5


dict_class={0:'background', 1:'roads', 2:'buildings'}
n_class = len(dict_class)

if __name__=='__main__':
    ret, ref_img = load_img(ref_file, grayscale=True)
    if ret !=0:
        print("Open file failed: {}".format(ref_file))
        sys.exit(-1)

    ret, pred_img = load_img(pred_file, grayscale=True)
    if ret != 0:
        print("Open file failed: {}".format(pred_file))
        sys.exit(-2)


    print("[INFO] Calculate confusion matrix..\n")

    height, width = ref_img.shape
    print(height, width)
    if height !=pred_img.shape[0] or width!=pred_img.shape[1]:
        print("image sizes of reference and predicted are not equal!\n")

    img_length = height*width
    num_checkpoints = np.int(img_length*check_rate)
    print("{} points will be used to evaluate accuracy!\n".format(num_checkpoints))
    pos = random.sample(range(img_length),num_checkpoints)


    """reshape images from two to one dimension"""
    ref_img = np.reshape(ref_img, height*width)
    pred_img = np.reshape(pred_img, height * width)

    labels = ref_img[pos]

    """ignore nodata pixels"""
    valid_index =[]
    for tt in valid_labels:
        ind = np.where(labels ==tt)
        ind = list(ind)
        valid_index.extend(ind[0])

    valid_index.sort()
    valid_num_checkpoints=len(valid_index)
    # valid_ref=ref_img[valid_index]
    valid_ref = labels[valid_index]
    print("valid value in reference image: {}".format(np.unique(valid_ref)))

    ts = pred_img[pos]
    valid_pred = ts[valid_index]
    print("valid value in predicton image: {}".format(np.unique(valid_pred)))

    tmp = np.unique(valid_pred)
    for ss in tmp:
        assert(ss in valid_labels)

    """Test to find out where are labels in the confusion matrix"""
    # num_tmp = labels[labels==0].shape
    # print("label 0 : {}".format(num_tmp))
    # num_tmp = labels[labels == 1].shape
    # print("label 1 : {}".format(num_tmp))
    # num_tmp = labels[labels == 2].shape
    # print("label 2 : {}".format(num_tmp))

    # confus_matrix = tf.contrib.metrics.confusion_matrix(labels, predictions, n_class)
    confus_matrix = tf.contrib.metrics.confusion_matrix(valid_pred,valid_ref,n_class)
    with tf.Session() as sess:
        confus_matrix = sess.run(confus_matrix)
    print(confus_matrix)

    confus_matrix = np.array(confus_matrix)

    oa =0
    x_row_plus = []
    x_col_plus=[]
    x_diagonal=[]
    for i in range(n_class):
        oa += confus_matrix[i,i]
        x_diagonal.append(confus_matrix[i,i])
        row_sum=sum(confus_matrix[i,:])
        col_sum=sum(confus_matrix[:,i])
        x_row_plus.append(row_sum)
        x_col_plus.append(col_sum)


    print("x_row_plus:{}".format(x_row_plus))
    print("x_col_plus:{}".format(x_col_plus))
    x_row_plus = np.array(x_row_plus)
    x_col_plus = np.array(x_col_plus)
    x_diagonal = np.array(x_diagonal)
    x_total = sum(x_row_plus)
    OA_acc = oa/(sum(x_row_plus))
    print("OA:{}".format(OA_acc))
    tmp = x_col_plus*x_row_plus
    kappa = (x_total*sum(x_diagonal)-sum(x_col_plus*x_row_plus))/np.float(x_total*x_total-sum(x_col_plus*x_row_plus))

    print("Kappa:{}".format(kappa))

    acc_roads = x_diagonal[1]/x_row_plus[1]
    print("roads_accuracy= {}".format(acc_roads))
    acc_buildings = x_diagonal[2] / x_row_plus[2]
    print("buildings_accuracy= {}".format(acc_buildings))

    recall_roads = x_diagonal[1] / x_col_plus[1]
    print("roads_recall= {}".format(recall_roads))
    recall_buildings = x_diagonal[2] / x_col_plus[2]
    print("buildings_recall= {}".format(recall_buildings))
    gc.collect()











#coding:utf-8

import numpy as np
import cv2
import os
import gc
import sys
import random
import tensorflow as tf

from ulitities.base_functions import load_img_by_cv2

# seed = 1
# random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

valid_labels=[0,1,2]
# valid_labels=[0,1]
dict_class={0:'background', 1:'roads', 2:'buildings'}
# dict_class={0:'background', 1:'shuidao'}
n_class = len(dict_class)


# ref_file = '/home/omnisky/PycharmProjects/data/test/shuidao/GF2shuitian22_test_label2.png'
ref_file = '../../data/test/paper/label/yujiang_test_label.png'
# 1) jian11_test_label, 2) jiangyou_label, 3) yujiang_test_label,
# 4) cuiping_label, 5) shuangliu_1test_label, 6) tongchuan_test_label
# sample1_label, yushui22_test_label, lizhou_test_label,
#  ruoergai_52test_label, jianyang_label,
# pred_file = '/home/omnisky/PycharmProjects/data/predict/unet/sat_4bands/' \
#             'unet_notonehot_tongchuan_4bands_combined.png'
# pred_file = '../../data/test/paper/pred_256/combined/' \
#             'unet_multiclass_jian_4bands_combined.png'

pred_file = '../../data/test/paper/voted/' \
            'unet_yujiang_4bands_voted2.png'
check_rate=1.0




if __name__=='__main__':
    ret, ref_img = load_img_by_cv2(ref_file, grayscale=True)
    if ret !=0:
        print("Open file failed: {}".format(ref_file))
        sys.exit(-1)

    ret, pred_img = load_img_by_cv2(pred_file, grayscale=True)
    print(np.unique(pred_img))
    if ret != 0:
        print("Open file failed: {}".format(pred_file))
        sys.exit(-2)

    print("\nfile: {}".format(os.path.split(pred_file)[1]))


    print("[INFO] Calculate confusion matrix..\n")

    height, width = ref_img.shape
    print(height, width)
    if height !=pred_img.shape[0] or width!=pred_img.shape[1]:
        print("image sizes of reference and predicted are not equal!\n")

    img_length = height*width
    assert(check_rate>0.001 and check_rate<=1.00)
    num_checkpoints = np.int(img_length*check_rate)

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
    print("{}points have been selected, but {} points will be used to evaluate accuracy!\n".format(num_checkpoints, valid_num_checkpoints))
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
    print("\nOA:{:.3f}".format(OA_acc))
    tmp = x_col_plus*x_row_plus
    kappa = (x_total*sum(x_diagonal)-sum(x_col_plus*x_row_plus))/np.float(x_total*x_total-sum(x_col_plus*x_row_plus))

    print("Kappa:{:.3f}".format(kappa))

    for i in range(n_class-1):
        i = i+1
        prec = x_diagonal[i] / x_row_plus[i]
        print("\n{}_accuracy= {:.3f}".format(dict_class[i], prec))
        recall = x_diagonal[i] / x_col_plus[i]
        print("{}_recall= {:.3f}".format(dict_class[i], recall))
        iou = x_diagonal[i] / (x_row_plus[i] + x_col_plus[i] - x_diagonal[i])
        print("{}_iou {:.3f}".format(dict_class[i], iou))

    # acc_roads = x_diagonal[1]/x_row_plus[1]
    # print("\nroads_accuracy= {:.3f}".format(acc_roads))
    # recall_roads = x_diagonal[1] / x_col_plus[1]
    # print("roads_recall= {:.3f}".format(recall_roads))
    # iou_roads = x_diagonal[1]/(x_row_plus[1]+x_col_plus[1]-x_diagonal[1])
    # print("roads_iou {:.3f}".format(iou_roads))
    #
    # acc_buildings = x_diagonal[2] / x_row_plus[2]
    # print("\nbuildings_accuracy= {:.3f}".format(acc_buildings))
    # recall_buildings = x_diagonal[2] / x_col_plus[2]
    # print("buildings_recall= {:.3f}".format(recall_buildings))
    # iou_buildings = x_diagonal[2] / (x_row_plus[2] + x_col_plus[2] - x_diagonal[2])
    # print("buildings_iou {:.3f}".format(iou_buildings))

    gc.collect()











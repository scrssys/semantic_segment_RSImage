#coding:utf-8

import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ulitities.base_functions import load_img_by_cv2, compare_two_image_size

ref_file ='../../data/test/paper/label/yujiang_test_label.png'
# 1) jian11_test_label, 2) jiangyou_label, 3) yujiang_test_label,
# 4) cuiping_label, 5) shuangliu_1test_label, 6) tongchuan_test_label

pred_file ='../../data/test/paper/voted/' \
            'unet_yujiang_4bands_voted0.png'

output_file = '../../data/test/paper/voted/' \
            'unet_yujiang_4bands_error_map.png'


if __name__=='__main__':
    print("[INFO]Reading images")
    ret, ref_img = load_img_by_cv2(ref_file, grayscale=True)
    if ret !=0:
        print("Open file failed:{}".format(ref_file))
        sys.exit(-1)

    ret, pred_img = load_img_by_cv2(pred_file, grayscale=True)
    if ret !=0:
        print("Open file failed:{}".format(pred_file))
        sys.exit(-2)

    compare_two_image_size(ref_img, pred_img, grayscale=True)

    height, width = ref_img.shape
    print("height,width: {},{}".format(height, width))

    match_img = np.zeros((height, width), np.uint8)

    for j in tqdm(range(height)):
        for i in range(width):
            if ref_img[j,i]!=0:
                if pred_img[j,i]==0:
                    match_img[j,i]=3  # 漏检的目标
                if ref_img[j,i]==pred_img[j,i]:
                    match_img[j,i]=2 # true positive 检测正确的目标
                elif ref_img[j,i]==1 and pred_img[j,i]==2:
                    match_img[j,i]=4 # 道路被错分为房屋建筑
                elif ref_img[j,i]==2 and pred_img[j,i]==1:
                    match_img[j,i]=5 # 房屋建筑被错分为道路
            else:
                if pred_img[j,i]!=0:
                    match_img[j,i]=1 # false negative 背景被错分为目标
                #ref_img[j,i]=pred_img[j,i]=0 # true negative

            #         match_img[j,i]=1
            # elif ref_img[j,i]==1:
            #     if pred_img[j,i]==1:
            #         match_img[j,i]=2
            #     else:
            #         match_img[j,i]=3
            # else:
            #     if pred_img[j,i]==2:
            #         match_img[j,i]=2
            #     else:
            #         match_img[j,i]=3

    plt.imshow(match_img)
    plt.show()

    cv2.imwrite(output_file, match_img)
    print("Saving into: {}".format(output_file))





    
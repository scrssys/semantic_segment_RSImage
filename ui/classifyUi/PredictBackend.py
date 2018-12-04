import cv2
import numpy as np
import os
import sys
import gc
import argparse
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras.preprocessing.image import img_to_array

from keras import backend as K
K.set_image_dim_ordering('tf')
K.clear_session()

from base_predict_functions import orignal_predict_notonehot, orignal_predict_onehot, smooth_predict_for_binary_notonehot, smooth_predict_for_binary_onehot, smooth_predict_for_multiclass
from ulitities.base_functions import load_img_normalization_by_cv2, load_img_by_gdal, UINT10,UINT8,UINT16
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands
from ulitities.base_functions import get_file

UINT8=0
UINT10 =1
UINT16=2

predictBinary_dict={'image_file':'', 'model_file':'', 'mask_path': '', 'im_bands':3, 'dtype':0,
                           'windsize':256, 'target_class':'roads', 'GPUID':'0', 'strategy':0}

dict_target={0: 'roads', 1: 'buildings'}

def predict_binary_for_single_image(input_dict={}):
    gup_id = input_dict['GPUID']
    os.environ["CUDA_VISIBLE_DEVICES"] = gup_id
    window_size = input_dict['windsize']
    im_bands = input_dict['im_bands']
    im_type = input_dict['dtype']
    FLAG_APPROACH_PREDICT = input_dict['strategy']
    img_file = input_dict['image_file']
    model_file = input_dict['model_file']
    output_file = input_dict['mask_path']

    out_bands = 1
    FLAG_ONEHOT = 0
    if input_dict['onehot']:
        FLAG_ONEHOT = 1


    input_img = load_img_by_gdal(img_file)
    if im_type == UINT8:
        input_img = input_img / 255.0
    elif im_type == UINT10:
        input_img = input_img / 1024.0
    elif im_type == UINT16:
        input_img = input_img / 65535.0

    input_img = np.clip(input_img, 0.0, 1.0)
    input_img = input_img.astype(np.float16)  # test accuracy

    """checke model file"""
    print("model file: {}".format(model_file))
    if not os.path.isfile(model_file):
        print("model does not exist:{}".format(model_file))
        sys.exit(-2)

    model = load_model(model_file)

    if FLAG_APPROACH_PREDICT==0:
        print("[INFO] predict image by orignal approach\n")
        if FLAG_ONEHOT:
            result = orignal_predict_onehot(input_img, im_bands, model, window_size)
        else:
            result = orignal_predict_notonehot(input_img,im_bands, model, window_size)
        print("result save as to: {}".format(output_file))
        cv2.imwrite(output_file, result*128)

    elif FLAG_APPROACH_PREDICT==1:
        print("[INFO] predict image by smooth approach\n")
        if FLAG_ONEHOT:
            result = predict_img_with_smooth_windowing_multiclassbands(
                input_img,
                model,
                window_size=window_size,
                subdivisions=2,
                real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
                pred_func=smooth_predict_for_binary_onehot
            )
        else:
            result = predict_img_with_smooth_windowing_multiclassbands(
                input_img,
                model,
                window_size=window_size,
                subdivisions=2,
                real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
                pred_func=smooth_predict_for_binary_notonehot
            )


        print("result save as to: {}".format(output_file))

        cv2.imwrite(output_file, result)
        print("Saved to {}".format(output_file))

    gc.collect()

    return 0


def predict_multiclass_for_single_image(input_dict={}):
    gup_id = input_dict['GPUID']
    os.environ["CUDA_VISIBLE_DEVICES"] = gup_id
    window_size = input_dict['windsize']
    im_bands = input_dict['im_bands']
    im_type = input_dict['dtype']
    FLAG_APPROACH_PREDICT = input_dict['strategy']
    img_file = input_dict['image_file']
    model_file = input_dict['model_file']
    output_dir = input_dict['mask_dir']

    out_bands = input_dict['target_num']

    input_img = load_img_by_gdal(img_file)
    if im_type == UINT8:
        input_img = input_img / 255.0
    elif im_type == UINT10:
        input_img = input_img / 1024.0
    elif im_type == UINT16:
        input_img = input_img / 65535.0

    input_img = np.clip(input_img, 0.0, 1.0)
    input_img = input_img.astype(np.float16)  # test accuracy

    abs_filename = os.path.split(img_file)[1]
    abs_filename = abs_filename.split(".")[0]
    print(abs_filename)

    """checke model file"""
    print("model file: {}".format(model_file))
    if not os.path.isfile(model_file):
        print("model does not exist:{}".format(model_file))
        sys.exit(-2)

    model = load_model(model_file)

    if FLAG_APPROACH_PREDICT==0:
        print("[INFO] predict image by orignal approach\n")
        result = orignal_predict_onehot(input_img, im_bands, model, window_size)
        output_file = ''.join([output_dir, '/', abs_filename, '.png'])
        print("result save as to: {}".format(output_file))
        cv2.imwrite(output_file, result)
        #
        # for b in range(out_bands):
        #     output_file = ''.join([output_dir, '/', abs_filename, '_', dict_target[b], '.png'])
        #     print("result save as to: {}".format(output_file))
        #     cv2.imwrite(output_file, result[:, :, b])

    elif FLAG_APPROACH_PREDICT==1:
        print("[INFO] predict image by smooth approach\n")
        result = predict_img_with_smooth_windowing_multiclassbands(
            input_img,
            model,
            window_size=window_size,
            subdivisions=2,
            real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
            pred_func=smooth_predict_for_multiclass
        )

        for b in range(out_bands):
            output_file = ''.join([output_dir,'/', abs_filename, '_', dict_target[b], '.png'])
            print("result save as to: {}".format(output_file))
            cv2.imwrite(output_file, result[:,:,b])

    gc.collect()

    return 0



def predict_binary_for_batch_image(input_dict={}):
    gup_id = input_dict['GPUID']
    os.environ["CUDA_VISIBLE_DEVICES"] = gup_id
    window_size = input_dict['windsize']
    im_bands = input_dict['im_bands']
    im_type = input_dict['dtype']
    FLAG_APPROACH_PREDICT = input_dict['strategy']
    input_path = input_dict['image_dir']
    model_file = input_dict['model_file']
    output_path = input_dict['mask_dir']

    out_bands = 1
    FLAG_ONEHOT = 0
    if input_dict['onehot']:
        FLAG_ONEHOT = 1

    all_files, num = get_file(input_path)
    if num == 0:
        print("There is no file in path:{}".format(input_path))
        sys.exit(-1)

    for img_file in all_files:
        print("[INFO] opening image...")
        print("FileName:{}".format(img_file))
        input_img = load_img_by_gdal(img_file)
        if im_type == UINT8:
            input_img = input_img / 255.0
        elif im_type == UINT10:
            input_img = input_img / 1024.0
        elif im_type == UINT16:
            input_img = input_img / 65535.0

        input_img = np.clip(input_img, 0.0, 1.0)
        input_img = input_img.astype(np.float16)

        model = load_model(model_file)

        abs_filename = os.path.split(img_file)[1]
        abs_filename = abs_filename.split(".")[0]

        if FLAG_APPROACH_PREDICT == 0:
            print("[INFO] predict image by orignal approach\n")
            if FLAG_ONEHOT:
                result = orignal_predict_onehot(input_img, im_bands, model, window_size)
            else:
                result = orignal_predict_notonehot(input_img, im_bands, model, window_size)

            output_file = ''.join([output_path, '/', abs_filename, '.png'])
            print("result save as to: {}".format(output_file))
            cv2.imwrite(output_file, result * 128)

        elif FLAG_APPROACH_PREDICT == 1:
            print("[INFO] predict image by smooth approach\n")
            if FLAG_ONEHOT:
                result = predict_img_with_smooth_windowing_multiclassbands(
                    input_img,
                    model,
                    window_size=window_size,
                    subdivisions=2,
                    real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
                    pred_func=smooth_predict_for_binary_onehot,
                    PLOT_PROGRESS=False
                )
            else:
                result = predict_img_with_smooth_windowing_multiclassbands(
                    input_img,
                    model,
                    window_size=window_size,
                    subdivisions=2,
                    real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
                    pred_func=smooth_predict_for_binary_notonehot,
                    PLOT_PROGRESS=False
                )

            output_file = ''.join([output_path, '/', abs_filename, 'smooth.png'])
            cv2.imwrite(output_file, result)
            print("Saved to: {}".format(output_file))

        gc.collect()

    return 0


def predict_multiclass_for_batch_image(input_dict={}):
    gup_id = input_dict['GPUID']
    os.environ["CUDA_VISIBLE_DEVICES"] = gup_id
    window_size = input_dict['windsize']
    im_bands = input_dict['im_bands']
    im_type = input_dict['dtype']
    FLAG_APPROACH_PREDICT = input_dict['strategy']
    input_path = input_dict['image_dir']
    model_file = input_dict['model_file']
    output_path = input_dict['mask_dir']

    out_bands = input_dict['target_num']


    all_files, num = get_file(input_path)
    if num == 0:
        print("There is no file in path:{}".format(input_path))
        sys.exit(-1)

    for img_file in all_files:
        print("[INFO] opening image...".format(img_file))
        input_img = load_img_by_gdal(img_file)
        if im_type == UINT8:
            input_img = input_img / 255.0
        elif im_type == UINT10:
            input_img = input_img / 1024.0
        elif im_type == UINT16:
            input_img = input_img / 65535.0

        input_img = np.clip(input_img, 0.0, 1.0)
        input_img = input_img.astype(np.float16)

        model = load_model(model_file)

        abs_filename = os.path.split(img_file)[1]
        abs_filename = abs_filename.split(".")[0]

        if FLAG_APPROACH_PREDICT == 0:
            print("[INFO] predict image by orignal approach\n")
            result = orignal_predict_onehot(input_img, im_bands, model, window_size)
            output_file = ''.join([output_path, '/', abs_filename, '.png'])
            print("result save as to: {}".format(output_file))
            cv2.imwrite(output_file, result * 128)

        elif FLAG_APPROACH_PREDICT == 1:
            print("[INFO] predict image by smooth approach\n")
            result = predict_img_with_smooth_windowing_multiclassbands(
                input_img,
                model,
                window_size=window_size,
                subdivisions=2,
                real_classes=out_bands,  # output channels = 是真的类别，总类别-背景
                pred_func=smooth_predict_for_multiclass,
                PLOT_PROGRESS=False
            )

            for b in range(out_bands):
                output_file = ''.join([output_path, '/', abs_filename, '_', dict_target[b],'smooth.png'])
                cv2.imwrite(output_file, result[:,:,b])
                print("Saved to: {}".format(output_file))

        gc.collect()

    return 0




import os
import sys
import gdal
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import medfilt, medfilt2d
from skimage import exposure
import cv2
from tqdm import tqdm
from PyQt5.QtCore import QFileInfo, QDir, QCoreApplication, Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from sampleProduce.SampleGenCommon import Ui_Dialog_sampleGenCommon

from ulitities.xml_prec import generate_xml_from_dict, parse_xml_to_dict
from ulitities.base_functions import get_file, load_img_by_gdal

sampleGen_dict={'input_dir':'', 'output_dir':'', 'window_size':256, 'min':0, 'max':2, 'target_label':1, 'sample_num':5000, 'mode':'augment'}


class child_sampleGenCommon(QDialog, Ui_Dialog_sampleGenCommon):
    def __init__(self):
        super(child_sampleGenCommon,self).__init__()
        # self.label_targetLabel.setVisible(self, False)
        # self.spinBox_targetLabel.setVisible(self, False)
        self.Flag_binary=True

        self.setupUi(self)

    def slot_input(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_input.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)

    def slot_output(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_output.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)

    def slot_strategy_binary(self):
        self.label_targetLabel.setVisible(True)
        self.spinBox_targetLabel.setVisible(True)

    def slot_strategy_multiclass(self):
        self.label_targetLabel.setVisible(False)
        self.spinBox_targetLabel.setVisible(False)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)

        self.Flag_binary = self.radioButton_binary.isChecked()

        input_dict = sampleGen_dict
        input_dict['input_dir'] = self.lineEdit_input.text()
        input_dict['output_dir'] = self.lineEdit_output.text()
        input_dict['window_size']= self.spinBox_windsize.value()
        input_dict['min'] = self.spinBox_min_2.value()
        input_dict['max'] = self.spinBox_max_2.value()
        input_dict['target_label'] = self.spinBox_targetLabel.value()
        assert (self.spinBox_targetLabel.value()<=self.spinBox_max_2.value())
        input_dict['sample_num'] = self.spinBox_sampNum.value()
        st = self.checkBox.isChecked()
        if st ==True:
            input_dict['mode'] = 'augument'
        else:
            input_dict['mode'] = 'original'

        instance = SampleGenerate(input_dict)
        if self.radioButton_binary.isChecked():
            instance.produce_training_samples_binary()
        else:
            instance.produce_training_samples_multiclass()

        QMessageBox.information(self, "Prompt", self.tr("Sample produced!"))

        self.setWindowModality(Qt.NonModal)




class SampleGenerate():
    def __init__(self, input_dict ={}):
        self.input_dict = input_dict

    def rotate(self, xb, yb, angle):
        xb = np.transpose(xb, (1, 2, 0))
        xb = np.rot90(np.array(xb), k=angle)
        xb = np.transpose(xb, ((2, 0, 1)))

        yb = np.rot90(np.array(yb), k=angle)

        return xb, yb

    def add_noise(self, xb, width, height, dtype=1):
        if dtype == 1:
            noise_value = 255
        elif dtype == 2:
            noise_value = 1024
        else:
            noise_value = 65535

        tmp = np.random.random() / 20.0  # max = 0.05
        noise_num = int(tmp * width * height)
        for i in range(noise_num):
            temp_x = np.random.randint(0, xb.shape[1])
            temp_y = np.random.randint(0, xb.shape[2])
            xb[:, temp_x, temp_y] = noise_value
        return xb

    def data_augment(self, xb, yb, w, h, d_type=1):
        if np.random.random() < 0.25:
            assert (yb.shape[0] == yb.shape[1])
            assert (xb.shape[1] == xb.shape[2])
            xb, yb = self.rotate(xb, yb, 1)
        if np.random.random() < 0.25:
            xb, yb = self.rotate(xb, yb, 2)
        if np.random.random() < 0.25:
            assert (yb.shape[0] == yb.shape[1])
            assert (xb.shape[1] == xb.shape[2])
            xb, yb = self.rotate(xb, yb, 3)
        if np.random.random() < 0.25:
            xb = np.transpose(xb, (1, 2, 0))
            xb = np.fliplr(xb)  # flip an array horizontally
            xb = np.transpose(xb, (2, 0, 1))
            yb = np.fliplr(yb)
        if np.random.random() < 0.25:
            xb = np.transpose(xb, (1, 2, 0))
            xb = np.flipud(xb)  # flip an array vertically (up down directory)
            xb = np.transpose(xb, (2, 0, 1))
            yb = np.flipud(yb)

        if np.random.random() < 0.25:  # gamma adjust
            tmp = np.random.random() * 3
            xb = exposure.adjust_gamma(xb, tmp)

        if np.random.random() < 0.25:  # medium filtering
            xb = xb.astype(np.float32)
            xb = np.transpose(xb, (1, 2, 0))
            _, _, bands = xb.shape
            for i in range(bands):
                xb[:, :, i] = medfilt2d(xb[:, :, i], (3, 3))
            xb = np.transpose(xb, (2, 0, 1))
            xb = xb.astype(np.uint16)

        if np.random.random() < 0.2:
            xb = self.add_noise(xb, w, h, d_type)

        return xb, yb

    def produce_training_samples_binary(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max']+1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num = int(self.input_dict['sample_num'])

        image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        print("\n[INFO] produce samples---------------------")
        g_count = 0
        for label_file in tqdm(label_files):

            src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
            if not os.path.isfile(src_file):
                print("Have no file:".format(src_file))
                continue

            print("src file:{}".format(os.path.split(src_file)[1]))
            label_img = load_img_by_gdal(label_file, grayscale=True)
            # print("label_img: {}".format(np.unique(label_img)))
            label_img = label_img.astype(np.uint8)
            y, x = label_img.shape
            # print("label_img: {}".format(np.unique(label_img)))


            dataset = gdal.Open(src_file)
            if dataset == None:
                print("open failed!\n")
                continue

            Y_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            if (X_width != x and Y_height != y):
                print("label and source image have different size:".format(label_file))
                continue

            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            src_img = dataset.ReadAsArray(0, 0, X_width, Y_height)
            src_img = np.array(src_img)

            del dataset

            index = np.where(label_img == target_label)
            all_label = np.zeros((Y_height, X_width), np.uint8)
            all_label[index] = 1

            print(np.unique(all_label))
            # if no pixel in target value, ignore this label file
            tp = np.unique(all_label)
            # if tp[0]==0:
            #     print("no target value in {}".format(label_file))
            #     continue
            #
            if len(tp) < 2:
                print("Only one value {} in {}".format(tp, label_file))
                if tp[0] == 0:
                    print("no target value in {}".format(label_file))
                    continue

            count = 0
            while count < image_each:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, Y_height - img_h - 1)
                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = all_label[random_height: random_height + img_h, random_width: random_width + img_w]

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue

                if 'augment' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, img_w, img_h, data_type)

                visualize = label_roi * 50

                cv2.imwrite((out_path + '/visualize/%d.png' % g_count), visualize)
                cv2.imwrite((out_path + '/label/%d.png' % g_count), label_roi)

                src_sample_file = out_path + '/src/%d.png' % g_count
                driver = gdal.GetDriverByName("GTiff")
                # driver = gdal.GetDriverByName("PNG")
                # outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, gdal.GDT_UInt16)
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1

    def produce_training_samples_multiclass(self):
        print('\ncreating dataset...')
        in_path = self.input_dict['input_dir']
        out_path = self.input_dict['output_dir']
        valid_labels = list(range(int(self.input_dict['min']), int(self.input_dict['max'] + 1)))
        target_label = int(self.input_dict['target_label'])

        label_files, tt = get_file(os.path.join(in_path, 'label/'))
        assert (tt != 0)

        image_num = int(self.input_dict['sample_num'])

        image_each = image_num / len(label_files)
        img_w = int(self.input_dict['window_size'])
        img_h = int(self.input_dict['window_size'])

        g_count = 0
        for label_file in tqdm(label_files):

            src_file = os.path.join(in_path, 'src/') + os.path.split(label_file)[1]
            if not os.path.isfile(src_file):
                print("Have no file:".format(src_file))
                continue
                # sys.exit(-1)

            print("src file:{}".format(os.path.split(src_file)[1]))

            label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            absname = os.path.split(label_file)[1]
            absname = absname.split('.')[0]

            dataset = gdal.Open(src_file)
            if dataset == None:
                print("open failed!\n")
                continue

            X_height = dataset.RasterYSize
            X_width = dataset.RasterXSize
            im_bands = dataset.RasterCount
            data_type = dataset.GetRasterBand(1).DataType

            # check size of label and src images
            x, y = label_img.shape
            print("Heigh, width of label is :{}, {}".format(x, y))
            print("Heigh, width of src is :{}, {}".format(X_height, X_width))
            if x != X_height or y != X_width:
                print("Warning: src and label have different size!")
                continue

            src_img = dataset.ReadAsArray(0, 0, X_width, X_height)
            src_img = np.array(src_img)

            del dataset

            count = 0
            while count < image_each:
                random_width = random.randint(0, X_width - img_w - 1)
                random_height = random.randint(0, X_height - img_h - 1)
                src_roi = src_img[:, random_height: random_height + img_h, random_width: random_width + img_w]
                label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

                """ignore nodata area"""
                FLAG_HAS_NODATA = False
                tmp = np.unique(label_img[random_height: random_height + img_h, random_width: random_width + img_w])
                for tt in tmp:
                    if tt not in valid_labels:
                        FLAG_HAS_NODATA = True
                        continue

                if FLAG_HAS_NODATA == True:
                    continue

                """ignore pure background area"""
                if len(np.unique(label_roi)) < 2:
                    if 0 in np.unique(label_roi):
                        continue
                # print(np.unique(label_roi))

                if 'augment' in self.input_dict['mode']:
                    src_roi, label_roi = self.data_augment(src_roi, label_roi, data_type)

                visualize = label_roi * 50

                cv2.imwrite((out_path + '/visualize/%d_%s.png' % (g_count,absname)), visualize)
                cv2.imwrite((out_path + '/label/%d_%s.png' % (g_count, absname)), label_roi)

                src_sample_file = out_path + '/src/%d_%s.png' % (g_count,absname)
                driver = gdal.GetDriverByName("GTiff")
                outdataset = driver.Create(src_sample_file, img_w, img_h, im_bands, data_type)
                if outdataset == None:
                    print("create dataset failed!\n")
                    sys.exit(-2)
                if im_bands == 1:
                    outdataset.GetRasterBand(1).WriteArray(src_roi)
                else:
                    for i in range(im_bands):
                        outdataset.GetRasterBand(i + 1).WriteArray(src_roi[i])
                del outdataset

                count += 1
                g_count += 1



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

from base_predict_functions import orignal_predict_notonehot, smooth_predict_for_binary_notonehot
from ulitities.base_functions import load_img_normalization_by_cv2, load_img_by_gdal, UINT10,UINT8,UINT16
from smooth_tiled_predictions import predict_img_with_smooth_windowing_multiclassbands

from PyQt5.QtCore import QFileInfo, QDir, QCoreApplication, Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from PredictBinaryForSingleimage import Ui_Dialog_predict_binary_single
from PredictMulticlassForSingleimage import Ui_Dialog_predict_multiclass_single
from PredictBinaryBatch import Ui_Dialog_predict_binary_batch
from PredictMulticlassBatch import Ui_Dialog_predict_multiclass_batch
from PredictBackend import predict_binary_for_single_image, predict_multiclass_for_single_image, predict_binary_for_batch_image, predict_multiclass_for_batch_image

predictBinary_dict={'image_file':'', 'model_file':'', 'mask_path': '', 'im_bands':3, 'dtype':0, 'onehot':False,
                           'windsize':256, 'target_class':'roads', 'GPUID':'0', 'strategy':'original'}
predictMulticlass_dict={'image_file':'', 'model_file':'', 'mask_dir': '', 'im_bands':3, 'dtype':0, 'target_num':2,
                           'windsize':256, 'GPUID':'0', 'strategy':'original'}

predictBinaryBatch_dict={'image_dir':'', 'model_file':'', 'mask_dir': '', 'im_bands':3, 'dtype':0, 'onehot':False,
                           'windsize':256, 'GPUID':'0', 'strategy':'original'}
predictMulticlassBatch_dict={'image_dir':'', 'model_file':'', 'mask_dir': '', 'im_bands':3, 'dtype':0, 'target_num':2,
                           'windsize':256, 'GPUID':'0', 'strategy':'original'}


class child_predictBinaryForSingleImage(QDialog, Ui_Dialog_predict_binary_single):
    def __init__(self):
        super(child_predictBinaryForSingleImage, self).__init__()
        self.setupUi(self)

    def slot_select_img_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select image', '../../data/', self.tr("img(*.png *.tif *.jpg)"))
        self.lineEdit_image.setText(str)
        # QDir.setCurrent(str)

    def slot_select_model_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select model', '../../data/', self.tr("model(*.h5)"))
        self.lineEdit_model.setText(str)

    def slot_save_mask_path(self):
        str, _ = QFileDialog.getSaveFileName(self, "Input mask saving path", '../../data/',
                                                 self.tr("mask(*.png *.jpg)"))
        self.lineEdit_mask.setText(str)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = predictBinary_dict

        input_dict['image_file'] = self.lineEdit_image.text()
        input_dict['model_file'] = self.lineEdit_model.text()
        input_dict['mask_path'] = self.lineEdit_mask.text()
        if not '.' in input_dict['mask_path']:
            input_dict['mask_path'] = ''.join([input_dict['mask_path'], '.png'])

        input_dict['im_bands'] = self.spinBox_bands.value()
        input_dict['dtype'] = self.comboBox_dtype.currentIndex()
        input_dict['windsize'] = self.spinBox_windsize.value()

        # input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()
        input_dict['strategy'] = self.comboBox_strategy.currentIndex()
        if self.checkBox.isChecked()==True:
            input_dict['onehot'] = True

        ret = -1
        ret = predict_binary_for_single_image(input_dict)
        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("Classify successfully!"))

        self.setWindowModality(Qt.NonModal)



class child_predictMulticlassForSingleImage(QDialog, Ui_Dialog_predict_multiclass_single):
    def __init__(self):
        super(child_predictMulticlassForSingleImage, self).__init__()
        self.setupUi(self)

    def slot_select_img_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select image', '../../data/', self.tr("img(*.png *.tif *.jpg)"))
        self.lineEdit_image.setText(str)
        # QDir.setCurrent(str)

    def slot_select_model_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select model', '../../data/', self.tr("model(*.h5)"))
        self.lineEdit_model.setText(str)

    def slot_save_mask_dir(self):
        str= QFileDialog.getExistingDirectory(self, "Select mask saving dir", '../../data/')
        self.lineEdit_mask_dir.setText(str)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = predictMulticlass_dict

        input_dict['image_file'] = self.lineEdit_image.text()
        input_dict['model_file'] = self.lineEdit_model.text()
        input_dict['mask_dir'] = self.lineEdit_mask_dir.text()
        input_dict['im_bands'] = self.spinBox_bands.value()
        input_dict['dtype'] = self.comboBox_dtype.currentIndex()
        input_dict['windsize'] = self.spinBox_windsize.value()

        # input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()
        input_dict['strategy'] = self.comboBox_strategy.currentIndex()
        input_dict['target_num'] = self.spinBox_classes.value()

        ret = -1
        ret = predict_multiclass_for_single_image(input_dict)
        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("Classify successfully!"))

        self.setWindowModality(Qt.NonModal)



class child_predictBinaryBatch(QDialog, Ui_Dialog_predict_binary_batch):
    def __init__(self):
        super(child_predictBinaryBatch, self).__init__()
        self.setupUi(self)

    def slot_select_img_dir(self):
        str = QFileDialog.getExistingDirectory(self, "Open image dir", '../../data/')
        self.lineEdit_images_dir.setText(str)
        # QDir.setCurrent(str)

    def slot_select_model_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select model', '../../data/', self.tr("model(*.h5)"))
        self.lineEdit_model.setText(str)

    def slot_save_mask_dir(self):
        str = QFileDialog.getExistingDirectory(self, "Select mask saving dir", '../../data/')
        self.lineEdit_mask_dir.setText(str)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = predictBinaryBatch_dict

        input_dict['image_dir'] = self.lineEdit_images_dir.text()
        input_dict['model_file'] = self.lineEdit_model.text()
        input_dict['mask_dir'] = self.lineEdit_mask_dir.text()
        input_dict['im_bands'] = self.spinBox_bands.value()
        input_dict['dtype'] = self.comboBox_dtype.currentIndex()
        input_dict['windsize'] = self.spinBox_windsize.value()

        # input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()
        input_dict['strategy'] = self.comboBox_strategy.currentIndex()
        if self.checkBox.isChecked()==True:
            input_dict['onehot'] = True

        ret = -1
        ret = predict_binary_for_batch_image(input_dict)
        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("Classify successfully!"))

        self.setWindowModality(Qt.NonModal)


class child_predictMulticlassBatch(QDialog, Ui_Dialog_predict_multiclass_batch):
    def __init__(self):
        super(child_predictMulticlassBatch, self).__init__()
        self.setupUi(self)

    def slot_select_img_dir(self):
        str = QFileDialog.getExistingDirectory(self, "Open image dir", '../../data/')
        self.lineEdit_images_dir.setText(str)
        # QDir.setCurrent(str)

    def slot_select_model_file(self):
        str, _ = QFileDialog.getOpenFileName(self, 'select model', '../../data/', self.tr("model(*.h5)"))
        self.lineEdit_model.setText(str)

    def slot_save_mask_dir(self):
        str = QFileDialog.getExistingDirectory(self, "Select mask saving dir", '../../data/')
        self.lineEdit_mask_dir.setText(str)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = predictMulticlassBatch_dict

        input_dict['image_dir'] = self.lineEdit_images_dir.text()
        input_dict['model_file'] = self.lineEdit_model.text()
        input_dict['mask_dir'] = self.lineEdit_mask_dir.text()
        input_dict['im_bands'] = self.spinBox_bands.value()
        input_dict['dtype'] = self.comboBox_dtype.currentIndex()
        input_dict['windsize'] = self.spinBox_windsize.value()

        # input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()
        input_dict['strategy'] = self.comboBox_strategy.currentIndex()
        input_dict['target_num'] = self.spinBox_classes.value()

        ret = -1
        ret = predict_multiclass_for_batch_image(input_dict)
        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("Classify successfully!"))

        self.setWindowModality(Qt.NonModal)


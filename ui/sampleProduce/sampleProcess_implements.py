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
from sampleProduce.SampleGenSelfAdapt import Ui_Dialog_sampleGenSelfAdapt

from ulitities.base_functions import get_file, load_img_by_gdal
from sampleProduce.sampleProcess_backend import SampleGenerate

sampleGen_dict={'input_dir':'', 'output_dir':'', 'window_size':256, 'min':0, 'max':2, 'target_label':1, 'sample_num':5000, 'mode':'augment'}
sampleGenSelfAdapt_dict={'input_dir':'', 'output_dir':'', 'window_size':256, 'min':0, 'max':2, 'target_label':1, 'sample_scaleRate':1.0, 'mode':'augment', 'imgmode':"normalize"}


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


class child_sampleGenSelfAdapt(QDialog, Ui_Dialog_sampleGenSelfAdapt):
    def __init__(self):
        super(child_sampleGenSelfAdapt,self).__init__()
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
        input_dict['min'] = self.spinBox_min.value()
        input_dict['max'] = self.spinBox_max.value()
        input_dict['target_label'] = self.spinBox_targetLabel.value()
        assert (self.spinBox_targetLabel.value()<=self.spinBox_max.value())
        input_dict['sample_scaleRate'] = self.doubleSpinBox_sampleScale.value()
        st = self.checkBox.isChecked()
        if st ==True:
            input_dict['mode'] = 'augument'
        else:
            input_dict['mode'] = 'original'

        sp = self.checkBox_normimg.isChecked()
        if sp == True:
            input_dict['imgmode'] = 'normalize'
        else:
            input_dict['imgmode'] = 'original'

        instance = SampleGenerate(input_dict)
        if self.radioButton_binary.isChecked():
            instance.produce_training_samples_binary_selfAdapt()
        else:
            instance.produce_training_samples_multiclass_selfAdapt()

        QMessageBox.information(self, "Prompt", self.tr("Sample produced!"))

        self.setWindowModality(Qt.NonModal)


#coding:utf-8

import os
import sys
import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PyQt5.QtWidgets import QDialog, QFileDialog
from ui.preProcess.image_stretch import Ui_Dialog_image_stretch



class child_image_stretch(QDialog, Ui_Dialog_image_stretch):
    def __init__(self):
        super(child_image_stretch,self).__init__()
        self.setupUi(self)

    def slot_select_input_dir(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_input.setText(dir_tmp)

    def slot_select_output_dir(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_output.setText(dir_tmp)

    def slot_ok(self):
        input_dir = self.lineEdit_input.text()
        output_dir = self.lineEdit_output.text()


# class ImageStretch():
#     def __init__(self):

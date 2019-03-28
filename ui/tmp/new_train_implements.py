import sys
import os

from keras.layers import *

from keras import backend as K
K.set_image_dim_ordering('tf')

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from TrainBinaryCommon import Ui_Dialog_train_binary_common
from tmp.new_train_backend import train_binary_for_ui

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)

trainBinary_dict={'trainData_path':'', 'saveModel_path':'', 'baseModel':'', 'im_bands':3, 'dtype':0,
                           'windsize':256, 'network':'unet', 'target_function':'crossentropy',
                  'class_name':'default', 'label_value': 1, 'BS':16, 'EPOCHS':100, 'GPUID':"0"}




class child_trainBinaryCommon(QDialog, Ui_Dialog_train_binary_common):
    def __init__(self):
        super(child_trainBinaryCommon, self).__init__()
        self.setupUi(self)

    def slot_traindatapath(self):
        str = QFileDialog.getExistingDirectory(self, "Train data path", '../../data/')
        self.lineEdit_traindata_path.setText(str)
        # QDir.setCurrent(str)

    def slot_savemodelpath(self):
        str = QFileDialog.getExistingDirectory(self, "Save model", '../../data/')
        self.lineEdit_savemodel.setText(str)

    def slot_basemodel(self):
        str, _= QFileDialog.getOpenFileName(self, "Select base model", '../../data/', self.tr("Models(*.h5)"))
        if not str=='':
            self.lineEdit_basemodel.setText(str)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = trainBinary_dict
        if os.path.isdir(self.lineEdit_traindata_path.text()):
            input_dict['trainData_path'] = self.lineEdit_traindata_path.text()
        if os.path.isdir(self.lineEdit_savemodel.text()):
            input_dict['saveModel_path'] = self.lineEdit_savemodel.text()
        if os.path.isfile(self.lineEdit_basemodel.text()):
            input_dict['baseModel'] = self.lineEdit_basemodel.text()

        input_dict['im_bands'] = int(self.spinBox_bands.value())
        input_dict['dtype'] = self.comboBox_dtype.currentIndex()
        input_dict['windsize'] = int(self.spinBox_windsize.value())
        if self.radioButton_unet.isChecked():
            input_dict['network'] = 'unet'
        elif self.radioButton_fcnnet.isChecked():
            input_dict['network'] = 'fcnnet'
        elif self.radioButton_segnet.isChecked():
            input_dict['network']='segnet'
        else:
            print("other network")
            sys.exit(-1)

        if self.radioButton_cross_entropy.isChecked():
            input_dict['target_function'] = 'crossentropy'
        elif self.radioButton_jaccard.isChecked():
            input_dict['target_function'] = 'jaccard'
        elif self.radioButton_jaccard_crossentropy.isChecked():
            input_dict['target_function']='jacc_and_cross'
        else:
            print("other function")
            sys.exit(-1)

        input_dict['class_name'] = self.lineEdit_class_name.text()
        input_dict['label_value'] = self.spinBox_label_value.value()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        ret =-1
        ret = train_binary_for_ui(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)
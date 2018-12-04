
import random
import sys
import os
import time
import matplotlib.pyplot as plt

from keras.layers import *
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, ReduceLROnPlateau

from keras import backend as K
K.set_image_dim_ordering('tf')

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from TrainBinaryJaccardCrossentropy import Ui_Dialog_train_binary_jaccCross
from TrainBinaryJaccard import Ui_Dialog_train_binary_jaccard
from TrainBinaryOnehot import Ui_Dialog_train_binary_onehot
from TrainBinaryCrossentropy import Ui_Dialog_train_binary_crossentropy
from TrainMulticlass import Ui_Dialog_train_multiclass
# from TrainBinaryCommon import Ui_Dialog_train_binary_common

from ulitities.base_functions import load_img_normalization
from train.semantic_segmentation_networks import binary_unet_jaccard, binary_fcnnet_jaccard, binary_segnet_jaccard
from modelTrainBackend import test_train, train_binary_jaccCross, train_binary_jaccard, train_binary_onehot, train_binary_crossentropy, train_multiclass


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)


trainBinary_dict={'trainData_path':'', 'saveModel_path':'', 'baseModel':'', 'im_bands':3, 'dtype':0,
                           'windsize':256, 'network':'unet', 'target_class':'roads', 'BS':16, 'EPOCHS':100, 'GPUID':"0"}

trainMulticlass_dict={'trainData_path':'', 'saveModel_path':'', 'baseModel':'', 'im_bands':3, 'dtype':0,
                           'windsize':256, 'network':'unet', 'classes':3, 'BS':16, 'EPOCHS':100, 'GPUID':"0"}



class child_trainBinaryJaccardCross(QDialog, Ui_Dialog_train_binary_jaccCross):
    def __init__(self):
        super(child_trainBinaryJaccardCross, self).__init__()
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
        input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        # instance = ModelTraining(input_dict)
        # instance.trainBinaryJaccCross()

        ret =-1
        ret = train_binary_jaccCross(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)


class child_trainBinaryJaccardOnly(QDialog, Ui_Dialog_train_binary_jaccard):
    def __init__(self):
        super(child_trainBinaryJaccardOnly, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("train jaccard")

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
        input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        # instance = ModelTraining(input_dict)
        # instance.trainBinaryJaccCross()

        ret =-1
        ret = train_binary_jaccard(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)


class child_trainBinaryOnehot(QDialog, Ui_Dialog_train_binary_onehot):
    def __init__(self):
        super(child_trainBinaryOnehot, self).__init__()
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
        input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        # instance = ModelTraining(input_dict)
        # instance.trainBinaryJaccCross()

        ret =-1
        ret = train_binary_onehot(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)

class child_trainBinaryCrossentropy(QDialog, Ui_Dialog_train_binary_crossentropy):
    def __init__(self):
        super(child_trainBinaryCrossentropy, self).__init__()
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
        input_dict['target_class'] = self.comboBox_target_class.currentText()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        # instance = ModelTraining(input_dict)
        # instance.trainBinaryJaccCross()

        ret =-1
        ret = train_binary_crossentropy(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)

class child_trainMulticlass(QDialog, Ui_Dialog_train_multiclass):
    def __init__(self):
        super(child_trainMulticlass, self).__init__()
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
        input_dict = trainMulticlass_dict
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
        input_dict['classes'] = self.spinBox_classes.value()
        input_dict['BS'] = self.spinBox_BS.value()
        input_dict['EPOCHS'] = self.spinBox_epoch.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        # instance = ModelTraining(input_dict)
        # instance.trainBinaryJaccCross()

        ret =-1
        ret = train_multiclass(input_dict)
        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("Model Traind successfully!"))


        self.setWindowModality(Qt.NonModal)





'''
class ModelTraining():
    def __init__(self, input_dict={}):
        self.input_dict = input_dict

        if os.path.isdir(input_dict['trainData_path']):
            self.train_data_path = input_dict['trainData_path']
        else:
            sys.exit(-2)

        date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print("date and time: {}".format(date_time))
        if os.path.isdir(input_dict['saveModel_path']):
            self.model_path = ''.join([input_dict['saveModel_path'], '/', input_dict['network'], '_',
                           input_dict['target_class'],'_binary_jaccard_', str(input_dict['windsize']),'_', date_time, '.h5'])
            self.saveModel_path = input_dict['saveModel_path']
        else:
            sys.exit(-3)

        self.base_model = input_dict['baseModel']
        if 'unet' in self.base_model:
            assert('unet' in self.network)
        elif 'fcnnet' in self.base_model:
            assert('fcnnet' in self.network)
        elif 'segnet' in self.base_model:
            assert('segnet' in self.base_model)

        self.network = input_dict['network']

        if self.input_dict['im_bands'] >0:
            self.im_bands = input_dict['im_bands']
        else:
            sys.exit(-4)
        self.im_type =input_dict['dtype']
        # if '8'in input_dict['dtype']:
        #     self.im_type = 'UINT8'
        # elif '10' in input_dict['dtype']:
        #     self.im_type = 'UINT10'
        # elif '16' in input_dict['dtype']:
        #     self.im_type = 'UINT16'
        # else:
        #     self.im_type = 'FLOAT'

        if input_dict['windsize']>32:
            self.img_w = input_dict['windsize']
            self.img_h = input_dict['windsize']
        if input_dict['BS']>1:
            self.BS = input_dict['BS']
        if input_dict['EPOCHS'] >1:
            self.EPOCHS = input_dict['EPOCHS']
        self.target_class = input_dict['target_class']
        if self.target_class not in self.train_data_path:
            print("target class and train data path is not consistent!")
            sys.exit(-5)
        self.gpu_id = input_dict['GPUID']

        self.n_label = 1


    """get the train file name and divide to train and val parts"""

    def get_train_val(self, val_rate=0.25):
        train_url = []
        train_set = []
        val_set = []
        for pic in os.listdir(self.train_data_path + '/src'):
            train_url.append(pic)
        random.shuffle(train_url)
        total_num = len(train_url)
        val_num = int(val_rate * total_num)
        for i in range(len(train_url)):
            if i < val_num:
                val_set.append(train_url[i])
            else:
                train_set.append(train_url[i])
        return train_set, val_set

    # data for training
    def generateData(self, batch_size,im_bands, im_type, train_data_path, img_w, img_h, n_label, data=[]):
        # print 'generateData...'

        while True:
            train_data = []
            train_label = []
            batch = 0
            for i in (range(len(data))):
                url = data[i]
                batch += 1
                _, img = load_img_normalization(self.im_bands, (self.train_data_path + '/src/' + url), data_type=self.im_type)

                # Adapt dim_ordering automatically
                img = img_to_array(img)
                train_data.append(img)
                _, label = load_img_normalization(1, (self.train_data_path + '/label/' + url))
                label = img_to_array(label)
                train_label.append(label)
                if batch % batch_size == 0:
                    # print 'get enough bacth!\n'
                    train_data = np.array(train_data)
                    train_label = np.array(train_label)
                    # train_label = to_categorical(train_label, num_classes=n_label)  # one_hot coding
                    train_label = train_label.reshape((batch_size, self.img_w * self.img_h, self.n_label))
                    yield (train_data, train_label)
                    train_data = []
                    train_label = []
                    batch = 0

    # data for validation
    def generateValidData(self, batch_size,im_bands, im_type, train_data_path, img_w, img_h, n_label,data=[]):
        # print 'generateValidData...'
        while True:
            valid_data = []
            valid_label = []
            batch = 0
            for i in (range(len(data))):
                url = data[i]
                batch += 1
                _, img = load_img_normalization(self.im_bands, (self.train_data_path + '/src/' + url), data_type=self.im_type)

                # Adapt dim_ordering automatically
                img = img_to_array(img)
                valid_data.append(img)
                _, label = load_img_normalization(1, (self.train_data_path + '/label/' + url))
                label = img_to_array(label)
                valid_label.append(label)
                if batch % batch_size == 0:
                    valid_data = np.array(valid_data)
                    valid_label = np.array(valid_label)
                    # valid_label = to_categorical(valid_label, num_classes=n_label)
                    valid_label = valid_label.reshape((batch_size, self.img_w * self.img_h, self.n_label))
                    yield (valid_data, valid_label)
                    valid_data = []
                    valid_label = []
                    batch = 0

    def trainBinaryJaccCross(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        if 'unet' in self.network:
            self.model = binary_unet_jaccard(self.img_w, self.img_h, self.im_bands, self.n_label)
        elif 'fcnnet' in self.network:
            self.model = binary_fcnnet_jaccard(self.img_w, self.img_h, self.im_bands, self.n_label)
        elif 'segnet' in self.network:
            self.model = binary_segnet_jaccard(self.img_w, self.img_h, self.im_bands, self.n_label)

        if os.path.isfile(self.base_model):
            print("load last weight from:{}".format(self.base_model))
            self.model.load_weights(self.base_model)

        model_checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_jaccard_coef_int',
            save_best_only=False)

        # model_checkpoint = ModelCheckpoint(
        #     model_save_path,
        #     monitor='val_jaccard_coef_int',
        #     save_best_only=True,
        #     mode='max')

        model_earlystop = EarlyStopping(
            monitor='val_jaccard_coef_int',
            patience=6,
            verbose=0,
            mode='max')

        """自动调整学习率"""
        model_reduceLR = ReduceLROnPlateau(
            monitor='val_jaccard_coef_int',
            factor=0.1,
            patience=3,
            verbose=0,
            mode='max',
            epsilon=0.0001,
            cooldown=0,
            min_lr=0
        )

        model_history = History()

        callable = [model_checkpoint, model_earlystop, model_reduceLR, model_history]
        # self.callable = [self.model_checkpoint,  self.model_history]
        # callable = [model_checkpoint,model_earlystop, model_history]
        train_set, val_set = self.get_train_val()
        train_numb = len(train_set)
        valid_numb = len(val_set)
        print("the number of train data is", train_numb)
        print("the number of val data is", valid_numb)

        H = self.model.fit_generator(
            generator=self.generateData(self.BS, self.im_bands, self.im_type, self.train_data_path, self.img_w,
                                        self.img_h, self.n_label, train_set),
            steps_per_epoch=train_numb // self.BS,
            epochs=self.EPOCHS,
            verbose=0,
            validation_data=self.generateValidData(self.BS, self.im_bands, self.im_type, self.train_data_path,
                                                   self.img_w, self.img_h, self.n_label, val_set),
            validation_steps=valid_numb // self.BS,
            callbacks=callable,
            max_q_size=1)

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = self.EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.plot(np.arange(0, N), H.history["jaccard_coef_int"], label="train_jaccard_coef_int")
        plt.plot(np.arange(0, N), H.history["val_jaccard_coef_int"], label="val_jaccard_coef_int")
        plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        fig_train_acc = ''.join([self.saveModel_path, self.network, '_',
                                 self.target, '_jaccard.png'])
        plt.savefig(fig_train_acc)
'''
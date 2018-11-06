import os
import sys
import gdal
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QMessageBox
from MainWin import Ui_MainWindow
from PyQt5.QtGui import QIcon
from preProcess.preprocess_implements import child_image_stretch, child_label, child_ImageClip
from sampleProduce.sampleProcess_implements import child_sampleGenCommon
from trainUi.trainModels_implements import child_trainBinaryJaccardCross, child_trainBinaryJaccardOnly, child_trainBinaryOnehot, child_trainBinaryCrossentropy, child_trainMulticlass
from classifyUi.predict_implements import child_predictBinaryForSingleImage, child_predictMulticlassForSingleImage, child_predictBinaryBatch, child_predictMulticlassBatch
from postProcess.postProcess_implements import child_CombineMulticlassFromSingleModelResults, child_VoteMultimodleResults, child_AccuacyEvaluate
from about import Ui_Dialog_about


class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.move(300,300)
        self.setWindowTitle(self.tr('Image'))
        self.setWindowIcon(QIcon('scrslogo.png'))
        self.setupUi(self)
        # self.new_translate()


    # def new_translate(self, ):
    #     _translate = QtCore.QCoreApplication.translate
    #     self.setWindowTitle(_translate(self, "image"))

    def for_action_label_check(self):
        child = child_label()
        child.show()
        child.exec_()

    def for_action_image_stretch(self):
        child = child_image_stretch()
        child.show()
        child.exec_()

    def slot_actiong_image_clip(self):
        child = child_ImageClip()
        child.show()
        child.exec_()

    def slot_action_sampleGenCommon(self):
        child = child_sampleGenCommon()
        child.show()
        child.exec_()

    def slot_action_trainBinaryJaccCross(self):
        child = child_trainBinaryJaccardCross()
        child.show()
        child.exec_()

    def slot_action_trainBinaryOnehot(self):
        child = child_trainBinaryOnehot()
        child.show()
        child.exec_()

    def slot_action_trainBinaryJaccard(self):
        child = child_trainBinaryJaccardOnly()
        child.show()
        child.exec_()

    def slot_action_trainBinaryCrossentropy(self):
        child = child_trainBinaryCrossentropy()
        child.show()
        child.exec_()


    def slot_action_trainMulticlass(self):
        child = child_trainMulticlass()
        child.show()
        child.exec_()

    def slot_action_predictBinarySingleImg(self):
        child = child_predictBinaryForSingleImage()
        child.show()
        child.exec_()

    def slot_action_predictMulticlassSingleImg(self):
        child = child_predictMulticlassForSingleImage()
        child.show()
        child.exec_()

    def slot_action_predictBinaryBatch(self):
        child = child_predictBinaryBatch()
        child.show()
        child.exec_()

    def slot_action_predictMulticlassBatch(self):
        child = child_predictMulticlassBatch()
        child.show()
        child.exec_()

    def slot_action_combineMulticlassFromSingleModel(self):
        child = child_CombineMulticlassFromSingleModelResults()
        child.show()
        child.exec_()

    def slot_action_VoteMultimodleResults(self):
        child = child_VoteMultimodleResults()
        child.show()
        child.exec_()

    def slot_action_accuracyEvaluate(self):
        child = child_AccuacyEvaluate()
        child.show()
        child.exec_()

    def slot_action_about(self):
        child = child_abount()
        child.show()
        child.exec_()


    def slot_open_show(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select image', '../../data/', self.tr("Image(*.png *.jpg *.tif)"))
        if not os.path.isfile(file):
            QMessageBox.warning(self, "Warning", 'Please select a raster image file!')
            sys.exit(-1)

        dataset = gdal.Open(file)
        if dataset == None:
            QMessageBox.warning(self, "Warning", 'Open file failed!')
            sys.exit(-2)
        im_band = dataset.RasterCount
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        data = dataset.ReadAsArray(0,0,width,height)
        data = np.array(data)

        if im_band ==1:
            plt.imshow(data, cmap='gray')
            plt.show()
        elif im_band ==3:
            data = data.transpose((1,2,0))
            plt.imshow(data)
            plt.show()
        elif im_band >3:
            data = data.transpose((1,2,0))
            img = data[:,:,:3]
            plt.imshow(data)
            plt.show()
        else:
            data = data.transpose((1, 2, 0))
            img = data[:, :, :0]
            plt.imshow(data)
            plt.show()



class child_abount(QDialog, Ui_Dialog_about):
    def __init__(self):
        super(child_abount, self).__init__()
        self.setupUi(self)


if __name__=='__main__':
    import sys
    app=QApplication(sys.argv)
    widget=mywindow()
    # widget = child_label()
    widget.show()
    sys.exit(app.exec_())

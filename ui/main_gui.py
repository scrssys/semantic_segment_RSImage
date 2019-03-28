import os
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
from postProcess.postProcess_implements import child_CombineMulticlassFromSingleModelResults, child_VoteMultimodleResults, child_AccuacyEvaluate, child_Binarization
from about import Ui_Dialog_about
from tmp.new_train_implements import child_trainBinaryCommon


class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.move(300,300)
        self.setWindowTitle(self.tr('Image'))
        self.setWindowIcon(QIcon('else/scrslogo.png'))
        self.setupUi(self)
        self.new_translate()


    def new_translate(self ):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "遥感影像人工地物深度学习识别系统"))
        self.menuFile.setTitle(_translate("MainWindow", "文件"))
        self.menuPrepocess.setTitle(_translate("MainWindow", "预处理"))
        self.menuTrain.setTitle(_translate("MainWindow", "模型训练"))
        self.menuClassify.setTitle(_translate("MainWindow", "分类识别"))
        self.menuHelp.setTitle(_translate("MainWindow", "帮助"))
        self.menuSampleProduce.setTitle(_translate("MainWindow", "数据集"))
        self.menuPostproc.setTitle(_translate("MainWindow", "后处理"))
        self.actionLabel_check.setText(_translate("MainWindow", "标注检查"))
        self.actionImage_strech.setText(_translate("MainWindow", "图像标准化"))
        self.actionSampleGenCommon.setText(_translate("MainWindow", "样本制作"))
        self.actionSampleGenByCV2.setText(_translate("MainWindow", "SampleGenByCV2"))
        self.actionImage_Clip.setText(_translate("MainWindow", "图像裁剪"))
        self.actionMismatch_Analyze.setText(_translate("MainWindow", "Mismatch Analyze"))
        self.actionTrain_Binary_Jaccard.setText(_translate("MainWindow", "二分类模型（Jaccard相似度）"))
        self.actionTrain_Binary_JaccCross.setText(_translate("MainWindow", "二分类模型（相似度&交叉熵）"))
        self.actionTrain_Binary_Cross_entropy.setText(_translate("MainWindow", "二分类模型（交叉熵）"))
        self.actionTrain_Multiclass.setText(_translate("MainWindow", "多分类模型"))
        self.actionTrain_Binary_Onehot_Cross.setText(_translate("MainWindow", "二分类模型（Onehot编码）"))
        self.actionPredict_Binary_Single.setText(_translate("MainWindow", "二分类预测"))
        self.actionPredict_Multiclass_Single.setText(_translate("MainWindow", "多分类预测"))
        self.actionPredict_Binary_Batch.setText(_translate("MainWindow", "二分类批处理"))
        self.actionPredict_Multiclass_Batch.setText(_translate("MainWindow", "多分类批处理"))
        self.actionAbout.setText(_translate("MainWindow", "关于"))
        self.actionOpen.setText(_translate("MainWindow", "影像打开"))
        self.actionExit.setText(_translate("MainWindow", "退出"))
        self.actionCombineSingleModelReults.setText(_translate("MainWindow", "多类别合成"))
        self.action_VoteMultiModelResults.setText(_translate("MainWindow", "多模型集成"))
        self.actionAccuracyEvaluation.setText(_translate("MainWindow", "精度评估"))
        self.actionBinarization.setText(_translate("MainWindow", "掩膜二值化"))

    def slot_action_binarization(self):
        child = child_Binarization()
        child.show()
        child.exec_()

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

    def slot_action_trainBinaryNew(self):
        child = child_trainBinaryCommon()
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



from PyQt5.QtCore import QFileInfo, QDir, QCoreApplication, Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from CombineMulticlassFromSingleModelResults import Ui_Dialog_combine_multiclas_fromsinglemodel
from VoteMultimodleResults import Ui_Dialog_vote_multimodels
from AccuracyEvaluate import Ui_Dialog_accuracy_evaluate
from Binarization import Ui_Dialog_binarization
from PostPrecessBackend import combine_masks, vote_masks, accuracy_evalute,binarize_mask,batchbinarize_masks


combinefile_dict = {'road_mask':'', 'building_mask':'', 'save_mask':'', 'foreground':127}
vote_dict = {'input_files':'', 'save_mask':'', 'target_values':[]}

accEvaluate_dict = {'gt_file':'', 'mask_file':'', 'valid_values':[], 'check_rate':0.5, 'GPUID':'5'}

binarization_dict = {'grayscale_mask':'', 'binary_mask':'', 'threshold':127}
binarybatch_dict = {'inputdir':'', 'outputdir':'', 'threshold':127}

class child_Binarization(QDialog, Ui_Dialog_binarization):

    def __init__(self):
        super(child_Binarization, self).__init__()
        self.setupUi(self)

    def slot_get_grayscale_mask(self):
        # str, _ = QFileDialog.getOpenFileName(self, "Select grayscale mask", '../../data/', self.tr("masks(*.png *jpg)"))
        # self.lineEdit_grayscale_mask.setText(str)
        # tp_dir = QFileInfo(str).path()
        # QDir.setCurrent(tp_dir)
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_grayscale_mask.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)


    def slot_get_saving_binary_mask_path(self):
        # str, _ = QFileDialog.getSaveFileName(self, "Save file to ...", '../../data/', self.tr("mask(*.png)"))
        # self.lineEdit_binary_mask.setText(str)
        # tp_dir = QFileInfo(str).path()
        # QDir.setCurrent(tp_dir)
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_binary_mask.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)


    def slot_ok(self):
        # self.setWindowModality(Qt.ApplicationModal)
        # input_dict = binarization_dict
        # input_dict['grayscale_mask'] = self.lineEdit_grayscale_mask.text()
        # input_dict['binary_mask'] = self.lineEdit_binary_mask.text()
        # input_dict['threshold'] = self.spinBox_forground.value()
        #
        # ret = -1
        # ret = binarize_mask(input_dict)
        #
        #
        # if ret ==0:
        #     QMessageBox.information(self,"Prompt", self.tr("successfully!"))
        # else:
        #     QMessageBox.warning(self, "Prompt", self.tr("Failed!"))
        #
        # self.setWindowModality(Qt.NonModal)

        self.setWindowModality(Qt.ApplicationModal)
        input_dict = binarybatch_dict
        input_dict['inputdir'] = self.lineEdit_grayscale_mask.text()
        input_dict['ouputdir'] = self.lineEdit_binary_mask.text()
        input_dict['threshold'] = self.spinBox_forground.value()

        ret = -1
        ret = batchbinarize_masks(input_dict)

        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("successfully!"))
        else:
            QMessageBox.warning(self, "Prompt", self.tr("Failed!"))

        self.setWindowModality(Qt.NonModal)



class child_CombineMulticlassFromSingleModelResults(QDialog, Ui_Dialog_combine_multiclas_fromsinglemodel):
    def __init__(self):
        super(child_CombineMulticlassFromSingleModelResults, self).__init__()
        self.setupUi(self)

    def slot_select_road_mask(self):
        str, _ = QFileDialog.getOpenFileName(self, "Select road mask", '../../data/', self.tr("masks(*.png *jpg)"))
        self.lineEdit_road_mask.setText(str)
        tp_dir = QFileInfo(str).path()
        QDir.setCurrent(tp_dir)

    def slot_select_building_mask(self):
        str, _ = QFileDialog.getOpenFileName(self, "Select building mask", '../../data/', self.tr("masks(*.png *jpg)"))
        self.lineEdit_building_mask.setText(str)
        tp_dir = QFileInfo(str).path()
        QDir.setCurrent(tp_dir)

    def slot_get_save_mask(self):
        str, _ = QFileDialog.getSaveFileName(self, "Save file", '../../data/', self.tr("mask(*.png)"))
        self.lineEdit_mask.setText(str)
        tp_dir = QFileInfo(str).path()
        QDir.setCurrent(tp_dir)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = combinefile_dict
        input_dict['road_mask'] = self.lineEdit_road_mask.text()
        input_dict['building_mask'] = self.lineEdit_building_mask.text()
        input_dict['save_mask'] = self.lineEdit_mask.text()
        if not '.png' in input_dict['save_mask']:
            input_dict['save_mask'] =''.join([input_dict['save_mask'], '.png'])
        input_dict['foreground'] = self.spinBox_forground.value()

        ret =-1
        ret = combine_masks(input_dict)

        if ret ==0:
            QMessageBox.information(self,"Prompt", self.tr("successfully!"))

        self.setWindowModality(Qt.NonModal)


class child_VoteMultimodleResults(QDialog, Ui_Dialog_vote_multimodels):
    def __init__(self):
        super(child_VoteMultimodleResults, self).__init__()
        self.setupUi(self)

    def slot_select_input_files(self):
        filelist, s = QFileDialog.getOpenFileNames(self, "Select files", '../../data/', self.tr("masks(*.png *jpg)"))
        filenum = len(filelist)
        str = self.lineEdit_inputs.text()
        if str != '':
            str += ';'
        for index, file in enumerate(filelist):
            if index ==filenum -1:
                str += file
            else:
                str +=file
                str +=';'
        self.lineEdit_inputs.setText(str)
        tp_dir = QFileInfo(filelist[0]).path()
        QDir.setCurrent(tp_dir)


    def slot_get_save_mask(self):
        str, _ = QFileDialog.getSaveFileName(self, "Save file", '../../data/', self.tr("mask(*.png)"))
        self.lineEdit_mask.setText(str)
        tp_dir = QFileInfo(str).path()
        QDir.setCurrent(tp_dir)


    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = vote_dict
        input_dict['input_files'] = self.lineEdit_inputs.text()
        input_dict['save_mask'] = self.lineEdit_mask.text()
        if not '.png' in input_dict['save_mask']:
            input_dict['save_mask'] =''.join([input_dict['save_mask'], '.png'])

        min = self.spinBox_min.value()
        max = self.spinBox_max.value()
        input_dict['target_values'] = list(range(min, max+1))

        ret =-1
        ret = vote_masks(input_dict)

        if ret ==0:
            QMessageBox.information(self, "Prompt", self.tr("successfully!"))

        self.setWindowModality(Qt.NonModal)


class child_AccuacyEvaluate(QDialog, Ui_Dialog_accuracy_evaluate):
    def __init__(self):
        super(child_AccuacyEvaluate, self).__init__()
        self.setupUi(self)

    def slot_select_gt_file(self):
        str, _ = QFileDialog.getOpenFileName(self, "Select ground-truth file", '../../data/', self.tr("mask(*.png *.tif)"))
        self.lineEdit_gt.setText(str)
        dir = QFileInfo(str).path()
        QDir.setCurrent(dir)


    def slot_select_mask_file(self):
        str, _ = QFileDialog.getOpenFileName(self, "Select mask file", '../../data/', self.tr("mask(*.png *.tif)"))
        self.lineEdit_mask.setText(str)
        tp_dir = QFileInfo(str).path()
        QDir.setCurrent(tp_dir)


    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        input_dict = accEvaluate_dict
        input_dict['gt_file'] = self.lineEdit_gt.text()
        input_dict['mask_file'] = self.lineEdit_mask.text()
        min = self.spinBox_min.value()
        max = self.spinBox_max.value()
        input_dict['valid_values'] = list(range(min, max+1))
        input_dict['check_rate'] = self.doubleSpinBox_rate.value()
        input_dict['GPUID'] = self.comboBox_gupid.currentText()

        ret =-1
        ret = accuracy_evalute(input_dict)

        if ret == 0:
            QMessageBox.information(self, "Prompt", self.tr("successfully!"))

        self.setWindowModality(Qt.NonModal)


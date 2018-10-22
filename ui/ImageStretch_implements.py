#coding:utf-8

import os
import sys
import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PyQt5.QtCore import QFileInfo, QDir, QCoreApplication, Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from ui.image_stretch import Ui_Dialog_image_stretch
from ulitities.xml_prec import generate_xml_from_dict
from ulitities.base_functions import get_file

imgStretch_dict = {'input_dir': '', 'output_dir': '', 'NoData': '65535', 'OutBits': '16bits',
                       'StretchRange': '1024',
                       'CutValue': '100'}

class child_image_stretch(QDialog, Ui_Dialog_image_stretch):
    def __init__(self):
        super(child_image_stretch,self).__init__()
        self.setWindowTitle("Image stretch")
        self.setupUi(self)

    def slot_select_input_dir(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_input.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)

    def slot_select_output_dir(self):
        dir_tmp = QFileDialog.getExistingDirectory(self, "select a existing directory", '../../data/')
        self.lineEdit_output.setText(dir_tmp)
        QDir.setCurrent(dir_tmp)

    def slot_ok(self):
        self.setWindowModality(Qt.ApplicationModal)
        imgStretch_dict['input_dir'] = self.lineEdit_input.text()
        imgStretch_dict['output_dir'] = self.lineEdit_output.text()
        nodata = self.spinBox_nodata.value()
        imgStretch_dict['NoData'] = str(nodata)
        imgStretch_dict['OutBits'] = self.comboBox_outbits.currentText()
        imgStretch_dict['stretchRange']=self.spinBox_range.value()
        imgStretch_dict['CutValue']=self.spinBox_cutvalue.value()

        # ss = QCoreApplication.applicationDirPath()
        QDir.setCurrent(QCoreApplication.applicationDirPath()) # change current dir to "venv/bin/"
        xmlfile = '../../metadata/image_stretch_inputs.xml'
        generate_xml_from_dict(imgStretch_dict, xmlfile)

        QMessageBox.information(self, 'Prompt', self.tr("Have saved the xml file !"))
        one_stretch = ImageStretch(imgStretch_dict)
        one_stretch.stretch_all_image()
        QMessageBox.information(self, 'Prompt', self.tr("Images stretched !"))
        self.setWindowModality(Qt.NonModal)



class ImageStretch():
    def __init__(self, inputDict, inputXml=''):
        self.in_dict = inputDict
        self.xmlfile = inputXml

    def stretch_all_image(self):
        if None == self.in_dict:
            QMessageBox.warning(self, "Warning", self.tr("input dict errors!"))
            sys.exit(-1)
        src_files, tt = get_file(self.in_dict['input_dir'])
        assert (tt != 0)
        NoData = int(self.in_dict['NoData'])
        valid_range = float(self.in_dict['StretchRange'])
        cut_value = float(self.in_dict['CutValue'])
        tp = self.in_dict['OutBits']
        FLAG_outbits = 0
        if '8' in self.in_dict['OutBits']:
            FLAG_outbits = 0
            assert(valid_range < 256)
        elif '16' in self.in_dict['OutBits']:
            FLAG_outbits = 1
            assert (valid_range < 65536)


        for file in tqdm(src_files):

            absname = os.path.split(file)[1]
            absname = absname.split('.')[0]
            # absname = 'shuidao.png'
            absname = ''.join([absname, '.png'])
            print(absname)
            if not os.path.isfile(file):
                print("input file dose not exist:{}\n".format(file))
                # sys.exit(-1)
                continue

            dataset = gdal.Open(file)
            if dataset == None:
                print("Open file failed: {}".format(file))
                continue

            height = dataset.RasterYSize
            width = dataset.RasterXSize
            im_bands = dataset.RasterCount
            im_type = dataset.GetRasterBand(1).DataType
            img = dataset.ReadAsArray(0, 0, width, height)
            del dataset
            # img = np.array(img, np.uint16)
            img = np.array(img, np.float32)
            result = []
            for i in range(im_bands):
                data = np.array(img[i])
                maxium = data.max()
                minm = data.min()
                mean = data.mean()
                std = data.std()
                print(maxium, minm, mean, std)
                data = data.reshape(height * width)
                ind = np.where((data > 0) & (data < NoData))
                ind = np.array(ind)

                a, b = ind.shape
                print("valid value number: {}\n".format(b))
                # tmp = np.zeros(b, np.uint16)
                tmp = np.zeros(b, np.float32)
                for j in range(b):
                    tmp[j] = data[ind[0, j]]
                tmaxium = tmp.max()
                tminm = tmp.min()
                tmean = tmp.mean()
                tstd = tmp.std()
                print(tmaxium, tminm, tmean, tstd)
                tt = (data - tmean) / tstd  # first Z-score normalization
                tt = (tt + 4) * valid_range / 8.0 - cut_value
                tind = np.where(data == 0)

                tt = np.array(tt)
                # tt = tt.astype(np.uint8)
                tt = tt.astype(np.uint16)
                tt[tind] = 0

                smaxium = tt.max()
                sminm = tt.min()
                smean = tt.mean()
                sstd = tt.std()
                print(smaxium, sminm, smean, sstd)

                out = tt.reshape((height, width))
                result.append(out)

            outputfile = os.path.join(self.in_dict['output_dir'], absname)
            driver = gdal.GetDriverByName("GTiff")

            if '8' in self.in_dict['OutBits']:
                outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_Byte)
            elif '16' in self.in_dict['OutBits']:
                outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)
            # outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)

            for i in range(im_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(result[i])

            del outdataset

        
